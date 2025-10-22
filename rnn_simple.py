#!/usr/bin/env python3
# Quick live demo (fast):
#   python rnn_simple.py --demo
#
# Full baseline (still simple):
#   python rnn_simple.py --epochs 10 --seq_len 32 --batch_size 128
#
# Force CPU for portability:
#   python rnn_simple.py --demo --device cpu

from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt


# -----------------------------
# Utilities
# -----------------------------

def set_seed(seed: int) -> None:
    """Seed python, numpy, and torch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device_arg: str) -> torch.device:
    if device_arg.lower() == "cpu":
        return torch.device("cpu")
    if device_arg.lower() == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class SplitData:
    X: np.ndarray  # (N, T, F)
    y: np.ndarray  # (N,)
    meta: Dict[str, np.ndarray]  # e.g., {"sim_id": (N,), "t_s": (N,)}


# -----------------------------
# Data utilities
# -----------------------------

def split_by_sim_id(df: pd.DataFrame, seed: int, demo: bool) -> Tuple[List[int], List[int], List[int]]:
    """Return lists of sim_id for train, val, test with 70/15/15 split.
    Shuffled deterministically.
    If demo=True, limit to first 20 groups after shuffle.
    """
    unique_ids = df["sim_id"].dropna().astype(int).unique().tolist()
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_ids)
    if demo:
        unique_ids = unique_ids[:20]
    n = len(unique_ids)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    train_ids = unique_ids[:n_train]
    val_ids = unique_ids[n_train:n_train + n_val]
    test_ids = unique_ids[n_train + n_val:]
    return train_ids, val_ids, test_ids


def build_sequences_for_ids(
    df: pd.DataFrame,
    sim_ids: List[int],
    seq_len: int,
    feature_cols: List[str],
) -> SplitData:
    """Build sliding-window sequences per provided sim_ids.

    For each group g sorted by t_s if present else index:
      X_t = rows [i - seq_len + 1 .. i] of features
      y_t = Vout at i+1
    Only valid when group length >= seq_len + 1.
    Returns arrays and aligned meta arrays for sim_id and t_s (target timestamp).
    """
    X_list: List[np.ndarray] = []
    y_list: List[float] = []
    sim_meta: List[int] = []
    t_meta: List[float] = []

    has_ts = "t_s" in df.columns

    for sid in sim_ids:
        g = df[df["sim_id"] == sid].copy()
        if g.empty:
            continue
        if has_ts:
            g = g.sort_values("t_s", kind="mergesort")
        else:
            g = g.sort_index(kind="mergesort")
        g = g.reset_index(drop=True)

        if len(g) < seq_len + 1:
            continue

        features = g[feature_cols].to_numpy(dtype=np.float32)
        targets = g["Vout"].to_numpy(dtype=np.float32)
        ts_vals = g["t_s"].to_numpy(dtype=np.float32) if has_ts else np.arange(len(g), dtype=np.float32)

        # Sliding windows: use features[i-seq_len+1:i+1] to predict targets[i+1]
        # Valid i range: seq_len-1 to len-2
        for i in range(seq_len - 1, len(g) - 1):
            X_window = features[i - seq_len + 1:i + 1]
            y_target = targets[i + 1]
            X_list.append(X_window)
            y_list.append(float(y_target))
            sim_meta.append(int(sid))
            t_meta.append(float(ts_vals[i + 1]))

    if len(X_list) == 0:
        return SplitData(
            X=np.zeros((0, seq_len, len(feature_cols)), dtype=np.float32),
            y=np.zeros((0,), dtype=np.float32),
            meta={"sim_id": np.zeros((0,), dtype=np.int64), "t_s": np.zeros((0,), dtype=np.float32)},
        )

    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.array(y_list, dtype=np.float32)
    meta = {
        "sim_id": np.array(sim_meta, dtype=np.int64),
        "t_s": np.array(t_meta, dtype=np.float32),
    }
    return SplitData(X=X, y=y, meta=meta)


class SeqDataset(Dataset):
    """Minimal dataset for (N, T, F) -> (N,) regression."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.X[idx]), torch.from_numpy(np.array(self.y[idx]))


class SimpleRNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        super().__init__()
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        out, h_n = self.rnn(x)  # out: (B, T, H), h_n: (num_layers, B, H)
        last_hidden = h_n[-1]   # (B, H)
        y = self.fc(last_hidden)  # (B, 1)
        return y.squeeze(-1)      # (B,)


# -----------------------------
# Training / Evaluation
# -----------------------------

def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    mse = nn.MSELoss()
    losses: List[float] = []
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        preds = model(xb)
        loss = mse(preds, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
    return float(np.mean(losses)) if losses else float("inf")


def evaluate_mse(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    mse = nn.MSELoss()
    losses: List[float] = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            preds = model(xb)
            loss = mse(preds, yb)
            losses.append(float(loss.detach().cpu().item()))
    return float(np.mean(losses)) if losses else float("inf")


def evaluate_test(model: nn.Module, loader: DataLoader, device: torch.device, meta: Dict[str, np.ndarray]) -> Tuple[float, pd.DataFrame]:
    model.eval()
    all_preds: List[np.ndarray] = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            preds = model(xb).detach().cpu().numpy()
            all_preds.append(preds)
    y_pred = np.concatenate(all_preds, axis=0) if all_preds else np.zeros((0,), dtype=np.float32)
    y_true = meta.get("y_true", None)
    if y_true is None:
        raise ValueError("Missing y_true in meta for test evaluation.")
    mae = float(mean_absolute_error(y_true, y_pred)) if len(y_true) > 0 else float("nan")
    out_df = pd.DataFrame({
        "sim_id": meta["sim_id"],
        "t_s": meta["t_s"],
        "y_true": y_true,
        "y_pred": y_pred,
    })
    return mae, out_df


# -----------------------------
# Main
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a simple vanilla RNN on cadence time-series to predict next-step Vout.")
    p.add_argument("--data_path", type=str, default="processed_waveforms.csv")
    p.add_argument("--seq_len", type=int, default=28)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden_size", type=int, default=64)
    p.add_argument("--num_layers", type=int, default=1)
    p.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--demo", action="store_true", help="Enable a fast demo mode with smaller data and epochs.")
    p.add_argument("--out_dir", type=str, default="outputs_simple")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Demo overrides
    if args.demo:
        args.epochs = 3
        args.seq_len = 16

    set_seed(args.seed)
    device = get_device(args.device)

    os.makedirs(args.out_dir, exist_ok=True)

    # Load data
    if not os.path.exists(args.data_path):
        print(f"Data not found at {args.data_path}")
        return
    df = pd.read_csv(args.data_path)

    # Select features: all numeric except Vout and sim_id; keep t_s if present
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in ["Vout", "sim_id"]]

    # Identify sim_id groups and split
    train_ids, val_ids, test_ids = split_by_sim_id(df, seed=args.seed, demo=args.demo)
    
    # Force sim_id=38 for test
    TARGET_SID = 38
    # Ensure 38 exists
    if TARGET_SID not in df["sim_id"].unique():
        raise ValueError(f"sim_id {TARGET_SID} not found in dataset.")
    # Remove 38 from train/val; make test exactly [38]
    train_ids = [sid for sid in train_ids if sid != TARGET_SID]
    val_ids   = [sid for sid in val_ids   if sid != TARGET_SID]
    test_ids  = [TARGET_SID]

    # Build sequences for splits
    train_data = build_sequences_for_ids(df, train_ids, args.seq_len, feature_cols)
    val_data = build_sequences_for_ids(df, val_ids, args.seq_len, feature_cols)
    test_data = build_sequences_for_ids(df, test_ids, args.seq_len, feature_cols)

    # Handle no valid sequences
    total_n = train_data.X.shape[0] + val_data.X.shape[0] + test_data.X.shape[0]
    if total_n == 0:
        print("No valid sequences found. Ensure groups have at least seq_len+1 rows.")
        return

    # Datasets / Loaders
    train_ds = SeqDataset(train_data.X, train_data.y)
    val_ds = SeqDataset(val_data.X, val_data.y)
    test_ds = SeqDataset(test_data.X, test_data.y)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    input_size = train_data.X.shape[2] if train_data.X.shape[0] > 0 else len(feature_cols)

    # Model / Optimizer
    model = SimpleRNN(input_size=input_size, hidden_size=args.hidden_size, num_layers=args.num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Train
    best_val = float("inf")
    best_path = os.path.join(args.out_dir, "best_rnn.pt")
    train_curve, val_curve = [], []
    for epoch in range(1, args.epochs + 1):
        train_mse = train_one_epoch(model, train_loader, optimizer, device)
        val_mse = evaluate_mse(model, val_loader, device)
        train_curve.append(train_mse)
        val_curve.append(val_mse)
        print(f"Epoch {epoch:03d} | Train MSE: {train_mse:.6f} | Val MSE: {val_mse:.6f}")
        if val_mse < best_val:
            best_val = val_mse
            torch.save(model.state_dict(), best_path)

    # Save training curve
    plt.figure(figsize=(7,4))
    plt.plot(train_curve, label="Train MSE")
    plt.plot(val_curve, label="Val MSE")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("Training/Validation MSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "loss_curve_mse.png"))
    plt.close()

    # Load best and evaluate on test
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))

    # For test evaluation, we want y_true aligned to loader order; since loader non-shuffled, order matches dataset
    test_meta = {
        "sim_id": test_data.meta["sim_id"],
        "t_s": test_data.meta["t_s"],
        "y_true": test_data.y,
    }
    test_mae, test_df = evaluate_test(model, test_loader, device, test_meta)
    
    # Compute MSE from saved predictions
    test_mse = float(np.mean((test_df["y_true"].to_numpy() - test_df["y_pred"].to_numpy())**2))
    
    print(f"[sim_id=38] Test MAE: {test_mae:.6f} | Test MSE: {test_mse:.6f} | seq_len={args.seq_len}")

    # Save predictions
    csv_path = os.path.join(args.out_dir, "test_preds.csv")
    test_df.to_csv(csv_path, index=False)

    # Plot sim_id=38
    if len(test_df) > 0:
        TARGET_SID = 38
        sub = test_df[test_df["sim_id"] == TARGET_SID].sort_values("t_s")
        if len(sub) > 0:
            plt.figure(figsize=(8, 4))
            plt.plot(sub["t_s"].to_numpy(), sub["y_true"].to_numpy(), label="True")
            plt.plot(sub["t_s"].to_numpy(), sub["y_pred"].to_numpy(), label="Pred", alpha=0.8)
            plt.xlabel("t_s")
            plt.ylabel("Vout")
            plt.title(f"Pred vs True Vout (sim_id={TARGET_SID})\nMAE={test_mae:.4e}  |  MSE={test_mse:.4e}")
            plt.legend()
            plt.tight_layout()
            suffix = "demo" if args.demo else "full"
            plot_name = f"pred_vs_true_sim38_{suffix}.png"
            plt.savefig(os.path.join(args.out_dir, plot_name))
            plt.close()
            
            # Residuals analysis
            res = sub["y_pred"].to_numpy() - sub["y_true"].to_numpy()

            # Residual vs time
            plt.figure(figsize=(8,3.8))
            plt.plot(sub["t_s"].to_numpy(), res)
            plt.xlabel("t_s")
            plt.ylabel("Residual (pred - true)")
            plt.title("Residuals vs Time (sim_id=38)")
            plt.tight_layout()
            plt.savefig(os.path.join(args.out_dir, "residuals_vs_time_sim38.png"))
            plt.close()

            # Residual histogram
            plt.figure(figsize=(6,3.8))
            plt.hist(res, bins=40)
            plt.xlabel("Residual")
            plt.ylabel("Count")
            plt.title("Residual Distribution (sim_id=38)")
            plt.tight_layout()
            plt.savefig(os.path.join(args.out_dir, "residual_hist_sim38.png"))
            plt.close()

    print(f"Saved best model to: {best_path}")
    print(f"Saved predictions to: {csv_path}")


if __name__ == "__main__":
    main()
