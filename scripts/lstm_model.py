#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
how to Use(run):
  python scripts/lstm_model.py \
    --csv processed_waveforms.csv \
    --epochs 20 \
    --hidden 128 \
    --layers 2 \
    --batch_size 32 \
    --sim_id 38
"""

import os, math, json, argparse, random, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")

#  Utilities 

def seed_all(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def masked_mse(pred, target, mask):
    # pred/target: [B,T,1], mask: [B,T]
    loss_elem = (pred - target) ** 2
    loss_elem = loss_elem.squeeze(-1)  # [B,T]
    return (loss_elem * mask).sum() / (mask.sum() + 1e-8)

def ensure_dirs(path):
    os.makedirs(path, exist_ok=True)

# Model 

class LSTMSeq2Seq(nn.Module):
    def __init__(self, in_dim, hidden=128, layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden,
            num_layers=layers,
            dropout=dropout if layers > 1 else 0.0,
            batch_first=True
        )
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)   # [B, T, H]
        yhat = self.head(out)   # [B, T, 1]
        return yhat

#  Main 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="processed_waveforms.csv",
                        help="Path to processed_waveforms.csv (preferred) or raw merged CSV.")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--sim_id", type=int, default=38,
                        help="Simulation ID to visualize (will be forced into test set if present).")
    parser.add_argument("--outdir", type=str, default="outputs",
                        help="Directory to save metrics/plots/series.")
    args = parser.parse_args()

    seed_all(42)
    ensure_dirs(args.outdir)

    #  Load data 
    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"CSV not found: {args.csv}")

    df = pd.read_csv(args.csv)
    required = {"sim_id", "t_s", "Vout"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must include columns: {required}")

    # Count waveforms (unique sim_id)
    n_waveforms = df["sim_id"].nunique()
    print(f"Unique waveforms (sim_id): {n_waveforms}")

    # Feature set (use what exists in your file)
    candidate_features = [
        "t_s", "Vdd", "tempC", "Cload_fF", "VinA", "VinB",
        "dVout_dt", "tau_f_est", "tau_r_est", "process", "gate"
    ]
    feature_cols = [c for c in candidate_features if c in df.columns]

    # Encode categoricals if present
    for cat in ["process", "gate"]:
        if cat in df.columns and not np.issubdtype(df[cat].dtype, np.number):
            df[cat] = df[cat].astype("category").cat.codes

    target_col = "Vout"

    # Order by time within each sim
    df = df.sort_values(["sim_id", "t_s"]).reset_index(drop=True)

    # Group & pad
    groups = [g for _, g in df.groupby("sim_id", sort=True)]
    sim_ids = [int(g["sim_id"].iloc[0]) for g in groups]
    max_len = max(len(g) for g in groups)

    scaler = StandardScaler().fit(df[feature_cols])

    def pad_group(g):
        X = scaler.transform(g[feature_cols].values).astype(np.float32)
        y = g[target_col].values.astype(np.float32).reshape(-1, 1)
        L = len(g)
        Xp = np.zeros((max_len, len(feature_cols)), dtype=np.float32)
        yp = np.zeros((max_len, 1), dtype=np.float32)
        mp = np.zeros((max_len,), dtype=np.float32)
        Xp[:L] = X
        yp[:L] = y
        mp[:L] = 1.0
        return Xp, yp, mp

    X_list, Y_list, M_list = [], [], []
    for g in groups:
        xp, yp, mp = pad_group(g)
        X_list.append(xp); Y_list.append(yp); M_list.append(mp)

    X = np.stack(X_list, axis=0)  # [N, T, F]
    Y = np.stack(Y_list, axis=0)  # [N, T, 1]
    M = np.stack(M_list, axis=0)  # [N, T]
    N, T, F = X.shape

    #  Train/Test split (force chosen sim_id into test) 
    chosen = args.sim_id if args.sim_id in sim_ids else sim_ids[0]
    indices = list(range(N))
    id_by_idx = {i: sim_ids[i] for i in indices}

    test_idx = [i for i in indices if id_by_idx[i] == chosen]
    remaining = [i for i in indices if i not in test_idx]
    rng = np.random.RandomState(42)
    # add ~20% of remaining to test
    extra = rng.choice(remaining, size=max(1, int(0.2 * len(remaining))), replace=False).tolist()
    test_idx += extra
    train_idx = [i for i in indices if i not in test_idx]

    X_train = torch.tensor(X[train_idx], dtype=torch.float32)
    Y_train = torch.tensor(Y[train_idx], dtype=torch.float32)
    M_train = torch.tensor(M[train_idx], dtype=torch.float32)

    X_test = torch.tensor(X[test_idx], dtype=torch.float32)
    Y_test = torch.tensor(Y[test_idx], dtype=torch.float32)
    M_test = torch.tensor(M[test_idx], dtype=torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMSeq2Seq(F, hidden=args.hidden, layers=args.layers, dropout=0.2).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_loader = DataLoader(TensorDataset(X_train, Y_train, M_train),
                              batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, Y_test, M_test),
                             batch_size=max(32, args.batch_size), shuffle=False)

    #  Train
    train_losses, val_losses = [], []
    for ep in range(1, args.epochs + 1):
        model.train()
        tl = 0.0
        for xb, yb, mb in train_loader:
            xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
            opt.zero_grad()
            yhat = model(xb)
            loss = masked_mse(yhat, yb, mb)
            loss.backward()
            opt.step()
            tl += loss.item() * xb.size(0)
        tl /= len(train_loader.dataset)
        train_losses.append(tl)

        # simple validation (masked MSE)
        model.eval()
        with torch.no_grad():
            vl = 0.0
            for xb, yb, mb in test_loader:
                xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
                yhat = model(xb)
                loss = masked_mse(yhat, yb, mb)
                vl += loss.item() * xb.size(0)
            vl /= len(test_loader.dataset)
            val_losses.append(vl)

        print(f"Epoch {ep:02d}/{args.epochs} | train_mMSE={tl:.6f} | val_mMSE={vl:.6f}")

    #  Evaluate across test set 
    model.eval()
    with torch.no_grad():
        yhat_test = model(X_test.to(device)).cpu().numpy()
    y_true = Y_test.numpy()
    m_np = M_test.numpy()

    mask_flat = m_np.reshape(-1) > 0
    y_true_flat = y_true.reshape(-1)[mask_flat]
    y_pred_flat = yhat_test.reshape(-1)[mask_flat]

    mae = mean_absolute_error(y_true_flat, y_pred_flat)
    mse_v = mean_squared_error(y_true_flat, y_pred_flat)
    rmse = math.sqrt(mse_v)
    mape = 100 * float(np.mean(np.abs((y_true_flat - y_pred_flat) / (np.abs(y_true_flat) + 1e-8))))
    smape = 100 * float(np.mean(2 * np.abs(y_pred_flat - y_true_flat) /
                                (np.abs(y_true_flat) + np.abs(y_pred_flat) + 1e-8)))
    r2 = r2_score(y_true_flat, y_pred_flat)
    try:
        r, _ = pearsonr(y_true_flat, y_pred_flat)
    except Exception:
        r = float("nan")

    metrics = {
        "csv": args.csv,
        "waveforms_total": int(n_waveforms),
        "num_features": int(F),
        "num_sims": int(N),
        "max_seq_len": int(T),
        "epochs": int(args.epochs),
        "hidden": int(args.hidden),
        "layers": int(args.layers),
        "batch_size": int(args.batch_size),
        "forced_test_sim": int(chosen),
        "test_sims": [int(sim_ids[i]) for i in test_idx],
        "MAE": float(mae),
        "MSE": float(mse_v),
        "RMSE": float(rmse),
        "MAPE_%": float(mape),
        "SMAPE_%": float(smape),
        "Forecast_Accuracy_% (100-MAPE)": float(100 - mape),
        "R2": float(r2),
        "Pearson_r": float(r)
    }

    # Save metrics 
    metrics_path = os.path.join(args.outdir, "lstm_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics: {metrics_path}")

    # Visualize chosen sim_id 
    # Get its global index (whole dataset order), then predict on that single sequence.
    if chosen in sim_ids:
        global_idx = sim_ids.index(chosen)
        with torch.no_grad():
            yhat_one = model(torch.tensor(X[global_idx:global_idx+1], dtype=torch.float32, device=device)).cpu().numpy()[0, :, 0]
        ytrue_one = Y[global_idx, :, 0]
        m_one = M[global_idx, :]
        # Pull original time axis for this sim
        t_axis = df[df["sim_id"] == chosen]["t_s"].values

        L = int(m_one.sum())
        t_trim = t_axis[:L] if len(t_axis) >= L else np.arange(L)
        ytrue_trim = ytrue_one[:L]
        yhat_trim = yhat_one[:L]

        # Save CSV series
        series_path = os.path.join(args.outdir, f"series_sim{chosen}.csv")
        pd.DataFrame({"t_s": t_trim, "Vout_true": ytrue_trim, "Vout_pred": yhat_trim}).to_csv(series_path, index=False)
        print(f"Saved series CSV: {series_path}")

        # Plot
        plt.figure(figsize=(9, 4.8))
        plt.plot(t_trim, ytrue_trim, label="True Vout")
        plt.plot(t_trim, yhat_trim, label="Predicted Vout", linestyle="--")
        plt.xlabel("t_s"); plt.ylabel("Vout")
        plt.title(f"LSTM Prediction — sim_id={chosen}")
        plt.legend()
        plot_path = os.path.join(args.outdir, f"plot_sim{chosen}.png")
        plt.savefig(plot_path, bbox_inches="tight"); plt.close()
        print(f"Saved plot: {plot_path}")
    else:
        print(f"sim_id {chosen} not found in data; skipped plot/series.")

    #  Also save training curve 
    curve_path = os.path.join(args.outdir, "training_curve.png")
    plt.figure(figsize=(8,4))
    plt.plot(train_losses, label="train masked MSE")
    plt.plot(val_losses, label="val masked MSE")
    plt.xlabel("epoch"); plt.ylabel("loss")
    plt.title("Training Curve (masked MSE)")
    plt.legend()
    plt.savefig(curve_path, bbox_inches="tight"); plt.close()
    print(f"Saved training curve: {curve_path}")

if __name__ == "__main__":
    main()
