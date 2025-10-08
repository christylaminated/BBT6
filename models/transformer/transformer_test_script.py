import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit

df = pd.read_csv("./processed_waveforms.csv")
cols_bool = ["gate_NAND2","gate_NOR2","process_SS","process_TT"]
X_num = df.drop(columns=["Vout"] + cols_bool).values
X_cat = df[cols_bool].apply(lambda x: x.astype('category').cat.codes).values

X = np.hstack([X_num, X_cat]).astype(np.float32)
y = df["Vout"].values

groups = df["sim_id"].values  # group by sim_id
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=groups))

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

class TransformerRegressor(nn.Module):
    def __init__(self, input_dim, embed_dim=64, num_heads=4, hidden_dim=128, num_layers=2):
        super().__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, 1)  # single output for regression

    def forward(self, x):
        x = self.embedding(x).unsqueeze(0)  # (1, batch, embed_dim)
        x = self.transformer(x)
        x = x.mean(dim=0)  # mean over sequence
        return self.fc(x)

input_dim = X_train.shape[1]
model = TransformerRegressor(input_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(10):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    print(f"Epoch [{epoch+1}/10], Loss: {loss.item():.4f}")

model.eval()
with torch.no_grad():
    preds = model(X_test)
    mse = criterion(preds, y_test)
    print(f"Test MSE: {mse.item():.4f}")


preds_np = preds.squeeze().detach().numpy()
output_df = pd.DataFrame({
    "sim_id": df.loc[test_idx, "sim_id"],
    "t_s": df.loc[test_idx, "t_s"],
    "Vout": preds_np
})
output_df = output_df.sort_values(by="sim_id", ascending=True)

# Save to CSV
output_df.to_csv("./models/transformer/predicted_Vout.csv", index=False)