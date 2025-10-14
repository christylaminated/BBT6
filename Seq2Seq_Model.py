# General Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# PyTorch Stuff
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os

FILE_PATH = 'processed_waveforms.csv'
SIM_ID = 'sim_id'
TIME = 't_s'
TARGET = 'Vout'

INPUT_SEQ_LEN = 50
OUTPUT_SEQ_LEN = 10

EPOCHS = 50
BATCH_SIZE = 32
# LEARNING_RATE = 0.001
LEARNING_RATE = 0.0005

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_data(file_path, input_seq_len, output_seq_len):
    df = pd.read_csv(file_path)

    bool_cols = ['gate_NAND2', 'gate_NOR2', 'process_SS', 'process_TT']

    for col in bool_cols:
        if col in df.columns and (df[col].dtype == 'object' or df[col].dtype == 'bool'):
            df[col] = df[col].apply(lambda x: 1 if str(x).lower() in ['true', '1'] else 0)

    feature_cols = [col for col in df.columns if col not in [SIM_ID, TIME]]
    feature_df = df[feature_cols]

    scaler = StandardScaler()
    scaler.fit(feature_df)
    
    target_scaler = StandardScaler()
    target_scaler.fit(df[[TARGET]])
    

    X, y = [], []
    
    for sim_id, group in df.groupby(SIM_ID):
        scaled_feat = scaler.transform(group[feature_cols])
        scaled_target = target_scaler.transform(group[[TARGET]])

        feature_np = np.array(scaled_feat, dtype = np.float32)
        target_np = np.array(scaled_target, dtype = np.float32)

        for i in range(len(group) - input_seq_len - output_seq_len + 1):
            X.append(feature_np[i : i + input_seq_len])
            y.append(target_np[i + input_seq_len : i + input_seq_len + 
                               output_seq_len])
            
    X, y = np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    return train_test_split(X, y, test_size = 0.2, random_state=123), scaler, target_scaler, df


# PyTorch Stuff
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers=1):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)

    def forward(self, x):
        _, (hidden, cell) = self.lstm(x)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, n_layers=1):
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(output_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden, cell):
        output, (hidden, cell) = self.lstm(x, (hidden, cell))
        predict = self.fc(output)
        
        return predict, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, target_len, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.target_len = target_len
        self.device = device

    def forward(self, source):
        batch_size = source.shape[0]
        # Makes sure that we're prediciting one Vout value for each time step
        target_dim = 1 
        hidden, cell = self.encoder(source)
        
        # Storing decoder output
        outputs = torch.zeros(batch_size, self.target_len, target_dim).to(self.device)
        decoder_input = torch.zeros(batch_size, 1, target_dim).to(self.device)
        
        for i in range(self.target_len):
            decoder_output, hidden, cell = self.decoder(decoder_input, hidden, cell)
            outputs[:, i, :] = decoder_output.squeeze(1)
            decoder_input = decoder_output
            
        return outputs
    
if __name__ == '__main__':
    (X_train, X_test, y_train, y_test), feature_scaler, target_scaler, original_df = load_data(
        FILE_PATH, INPUT_SEQ_LEN, OUTPUT_SEQ_LEN
    )

    # Set Up
    train_data = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_data = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    train_load = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_load = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    input_dim = X_train.shape[2]
    hidden_dim = 256
    
    encoder = Encoder(input_dim, hidden_dim, n_layers=2).to(DEVICE)
    decoder = Decoder(1, hidden_dim, n_layers=2).to(DEVICE)
    model = Seq2Seq(encoder, decoder, OUTPUT_SEQ_LEN, DEVICE).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training
    print("\n Starting Model Training ")
    for epoch in range(EPOCHS):
        model.train()
        t_loss = 0
        for x_batch, y_batch in train_load:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            
            optimizer.zero_grad()
            prediction = model(x_batch)

            weights = torch.ones_like(y_batch)
            weights[y_batch > 0.01] = 10.0
            loss = torch.mean(weights * (prediction - y_batch)**2)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            t_loss += loss.item()
        
        avg_loss = t_loss / len(train_load)
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.6f}")
    print(" Model Training Complete \n")

    #Evaluation Time
    model.eval()
    predict_scaled = []
    target_scaled = []
    with torch.no_grad():
        for x_batch, y_batch in test_load:
            x_batch = x_batch.to(DEVICE)
            prediction = model(x_batch)
            predict_scaled.append(prediction.cpu().numpy())
            target_scaled.append(y_batch.numpy())

    predict_scaled = np.concatenate(predict_scaled, axis=0)
    target_scaled = np.concatenate(target_scaled, axis=0)

    # Reshape for inverse scaling and metrics calculation
    num_samples = predict_scaled.shape[0] * predict_scaled.shape[1]
    predict_flat = predict_scaled.reshape(num_samples, 1)
    target_flat = target_scaled.reshape(num_samples, 1)

    # Inverse transform to get actual voltage values
    predict_unscaled = target_scaler.inverse_transform(predict_flat)
    target_unscaled = target_scaler.inverse_transform(target_flat)

    # Calculate metrics on the unscaled data
    mse = mean_squared_error(target_unscaled, predict_unscaled)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(target_unscaled, predict_unscaled)

    # Accuracy score based on that simulation
    ACCURACY_TOLERANCE_PERCENT = 5.0
    voltage_range = np.max(target_unscaled) - np.min(target_unscaled)
    tolerance_value = (ACCURACY_TOLERANCE_PERCENT / 100.0) * voltage_range
    
    absolute_errors = np.abs(predict_unscaled - target_unscaled)
    correct_predictions = np.sum(absolute_errors <= tolerance_value)
    accuracy = (correct_predictions / len(target_unscaled)) * 100


    print("--- Evaluation Metrics on Test Set ---")
    print(f"Mean Squared Error (MSE):      {mse:.6f}")
    print(f"Root Mean Squared Error (RMSE):  {rmse:.6f}")
    print(f"Mean Absolute Error (MAE):     {mae:.6f}")
    print(f"Accuracy (within +/- {tolerance_value:.4f}V): {accuracy:.2f}%")
    print("--------------------------------------\n")

# --- Full Simulation Visualization ---
print("--- Generating Full Sim Plot ---")
FULL_DATA = 'full_waveforms.csv' 
MARKER = 10

model.eval()
with torch.no_grad():
    full_df = pd.read_csv(FULL_DATA)

    # Re-apply the same boolean conversions you did during training
    bool_cols = ['gate_NAND2', 'gate_NOR2', 'process_SS', 'process_TT']
    for col in bool_cols:
        if col in full_df.columns and (full_df[col].dtype == 'object' or full_df[col].dtype == 'bool'):
            full_df[col] = full_df[col].apply(lambda x: 1 if str(x).lower() in ['true', '1'] else 0)

    # sim_id_plot = 38
    sim_id_plot = 26
    sim_df = full_df[full_df[SIM_ID] == sim_id_plot].copy()
    
    print(f"Visualizing results for specified simulation (ID: {sim_id_plot})...")
    time_raw = sim_df[TIME].values
    time_axis = time_raw - time_raw.min()

    feature_col = [c for c in sim_df.columns if c not in [SIM_ID, TIME]]
    sim_scaled = feature_scaler.transform(sim_df[feature_col])
    
    sim_seq = []
    for i in range(len(sim_df) - INPUT_SEQ_LEN + 1):
        sim_seq.append(sim_scaled[i:i+INPUT_SEQ_LEN])
    
    sim_seq_np = np.array(sim_seq, dtype=np.float32)
    sim_seq_tensor = torch.from_numpy(sim_seq_np).to(DEVICE)
    
    sim_preds_scaled = []
    pred_loader = DataLoader(TensorDataset(sim_seq_tensor), batch_size=BATCH_SIZE)
    
    for batch in pred_loader:
        sim_preds_scaled.append(model(batch[0]).cpu().numpy())
    
    sim_preds_scaled = np.concatenate(sim_preds_scaled, axis=0)

    first_preds_scaled = sim_preds_scaled[:, 0, :]
    pred = target_scaler.inverse_transform(first_preds_scaled)

    #NEW
    pred = np.clip(pred, 0, None)

    gnd_truth = sim_df[TARGET].values

    plot_pred = np.full(len(gnd_truth), np.nan)
    start_index = INPUT_SEQ_LEN
    end_index = len(gnd_truth)
    num_plot = end_index - start_index
    plot_pred[start_index:end_index] = pred[:num_plot].flatten()

    # Figure Stuff
    plt.figure(figsize=(18, 8))

    plt.plot(time_axis, gnd_truth, linestyle='-', marker='o', color='blue', 
                label='Ground Truth Voltage', markersize=4, zorder=1, markevery=MARKER)
    
    plt.plot(time_axis, plot_pred, linestyle='--', marker='o', color='red', 
                label='Predicted Voltage', markersize=4, zorder=2, markevery=MARKER)
    
    plt.title(f"Voltage Prediction vs. Ground Truth (Simulation ID: {sim_id_plot})")
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (V)")
    plt.legend()
    plt.grid(True)
    plt.show()