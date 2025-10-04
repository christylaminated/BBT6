import torch, torch.nn as nn, torch.optim as optim
import pandas as pd, numpy as np
from sklearn.preprocessing import StandardScaler

# --- Load & preprocess ---
df = pd.read_csv("processed_waveform.csv")
feature_cols = ["t_s","Vdd","tempC","Cload_fF","VinA","VinB","dVout_dt","tau_f_est","tau_r_est"]
target_col = "Vout"

groups = [g for _, g in df.groupby("sim_id")]
max_len = max(len(g) for g in groups)

scaler = StandardScaler().fit(df[feature_cols])
def pad_group(g):
    X = scaler.transform(g[feature_cols])
    y = g[target_col].values
    Xp, yp = np.zeros((max_len, X.shape[1])), np.zeros(max_len)
    Xp[:len(X)], yp[:len(y)] = X, y
    mask = np.zeros(max_len, bool); mask[len(X):] = True
    return Xp, yp, mask

Xs, Ys, Ms = zip(*[pad_group(g) for g in groups])
X = torch.tensor(np.stack(Xs),dtype=torch.float32) # (B,L,F)
Y = torch.tensor(np.stack(Ys),dtype=torch.float32) # (B,L)
M = torch.tensor(np.stack(Ms))                     # (B,L)

# --- Model pieces ---
f, d, h, L = X.shape[2], 64, 4, 2
inp = nn.Linear(f, d)
enc_layer = nn.TransformerEncoderLayer(d_model=d, nhead=h, dim_feedforward=128, dropout=0.1, batch_first=True)
encoder = nn.TransformerEncoder(enc_layer, num_layers=L)
out = nn.Linear(d, 1)

# --- Forward pass fn ---
def forward(x, mask):
    x = inp(x) * d**0.5         # (B,L,F)->(B,L,d)
    x = encoder(x, src_key_padding_mask=mask)  # (B,L,d)
    return out(x).squeeze(-1)   # (B,L)

# --- Training ---
opt = optim.Adam(list(inp.parameters())+list(encoder.parameters())+list(out.parameters()), lr=1e-3)
lossf = nn.MSELoss(reduction="none")

for epoch in range(20):
    pred = forward(X, M)
    loss = lossf(pred, Y)
    loss = (loss.masked_fill(M,0).sum()/(~M).sum())  # ignore padded steps
    opt.zero_grad(); loss.backward(); opt.step()
    print(epoch, loss.item())
