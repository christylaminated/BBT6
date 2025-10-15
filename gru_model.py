import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error

#libraries used for TensorFlow
import tensorflow as tf
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences


import os

df = pd.read_csv('processed_waveforms.csv')
print("dataframe's shape:", df.shape)

#sorting the data using sim_id and
df = df.sort_values(by=['sim_id','t_s'])

#split like the big dataframes into smaller chunks (one per sim_id) for the model to understand
group_simIDs = [g for _, g in df.groupby('sim_id')]

# features and target output 
target_column = 'Vout'
feature_column = [col for col in df.columns if col not in ['sim_id', target_column]]

#convert to sequences -- which the GRU model will understand
X= [g[feature_column].values for g in group_simIDs]
y= [g[target_column].values for g in group_simIDs]

#using this line to check and see if padding is needed -- padding is needed if the sequences are not of equal length; in this case, it is not. 
print(df.groupby('sim_id')['t_s'].count().describe())

#padding by adding 0s to the one that aren't max
X = tf.keras.preprocessing.sequence.pad_sequences(X, dtype='float32', padding='post')
y = tf.keras.preprocessing.sequence.pad_sequences(y, dtype='float32', padding='post')

print("Padding X shape: ",X.shape)
print("Padding y shape: ",y.shape)


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state = 123)


# established 2 sequential layers here: 64 memory units for recent details; 32 memory units builds the bigger picture analysis

train_mask = np.expand_dims((y_train != 0.0).astype(float), -1)
test_mask = np.expand_dims((y_test != 0.0).astype(float), -1)


GRU_model = tf.keras.Sequential([
 tf.keras.layers.Masking(mask_value=0.0,input_shape=(X.shape[1],X.shape[2]) ),
 tf.keras.layers.GRU(64,return_sequences=True), 
 tf.keras.layers.GRU(32,return_sequences=True),
 tf.keras.layers.Dense(1)
])

GRU_model.compile(optimizer='adam',loss='mse',metrics=['mae'])



voltage_analysis = GRU_model.fit(
 X_train, y_train,
 validation_data = (X_test,y_test),
 epochs = 50,
 batch_size= 32
)


#evaluate 
loss, mae = GRU_model.evaluate(X_test, y_test)
print(f"Test MAE: {mae: .4f}")

y_pred = GRU_model.predict(X_test[:1])[0]

non_padded = np.count_nonzero(y_test[0])

#plotting
plt.figure(figsize=(8,4))
plt.plot(y_test[0][:non_padded], label ='Actual')
plt.plot(y_pred[:non_padded], label='Predict')
plt.title("GRU Prediction vs. Actual")
plt.xlabel("Time(s)")
plt.ylabel("Vout")
plt.legend()
plt.show()


'''
# each sim = one waveform
X, y, t_values, sim_ids = [], [], [], []
for sim_id, group in df.groupby('sim_id'):
    group = group.sort_values(by='t_s')  #time sort
    X.append(group[feature_cols].values)
    y.append(group[target_col].values)
    t_values.append(group['t_s'].values)
    sim_ids.append(sim_id)

# padding the sequences because to make sure all the lengths of sim_id are same
X = pad_sequences(X, padding='post', dtype='float32')
y = pad_sequences(y, padding='post', dtype='float32')

print("Shape of X:", X.shape)
print("Shape of y:", y.shape)


num_sims = len(X)
indices = np.arange(num_sims)
np.random.seed(42)
np.random.shuffle(indices)

#split set
split_idx = int(0.8 * num_sims)
train_idx = indices[:split_idx]
test_idx = indices[split_idx:]

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]
t_train = [t_values[i] for i in train_idx]
t_test = [t_values[i] for i in test_idx]
sim_train = [sim_ids[i] for i in train_idx]
sim_test = [sim_ids[i] for i in test_idx]

print(f"Training on {len(train_idx)} simulations, Testing on {len(test_idx)} simulations")

#GRU Model
timesteps = X_train.shape[1]
features = X_train.shape[2]

model = Sequential([
    GRU(64, return_sequences=True, input_shape=(timesteps, features)),
    Dropout(0.2),
    GRU(32, return_sequences=True),
    Dense(1)  
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
model.summary()

# training set
history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=16,
    validation_split=0.2,
    verbose=1
)

# evaluate
loss, mae = model.evaluate(X_test, y_test)
print(f"Test Loss (MSE): {loss:.4f}, MAE: {mae:.4f}")

# plot shown
plt.figure(figsize=(8,5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('GRU Training & Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True)
plt.show()

# Predict and Plot VOUT vs TIME 
output_dir = 'gru_vout_vs_time_plots'
os.makedirs(output_dir, exist_ok=True)

y_pred = model.predict(X_test)
accuracy_threshold = 0.1

for i in range(min(5, len(X_test))):  
    true_vals = y_test[i].flatten()
    pred_vals = y_pred[i].flatten()
    time_vals = t_test[i].flatten()

    # trimming all the padded zeros
    valid_len = np.count_nonzero(time_vals)
    true_vals = true_vals[:valid_len]
    pred_vals = pred_vals[:valid_len]
    time_vals = time_vals[:valid_len]

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(time_vals, true_vals, 'b-', label='Actual Vout', linewidth=2)
    plt.plot(time_vals, pred_vals, 'r--', label='Predicted Vout', linewidth=2)
    plt.title(f'GRU Prediction: Vout vs Time (Simulation ID: {sim_test[i]})')
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (V)')
    plt.legend()
    plt.grid(True)

    # mse,rmse, mae and accuracy
    mse = mean_squared_error(true_vals, pred_vals)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true_vals, pred_vals)
    accuracy = np.mean(np.abs(true_vals - pred_vals) <= accuracy_threshold) * 100

    

   

    plt.tight_layout(rect=[0.15, 0, 1, 1])
    file_path = os.path.join(output_dir, f'vout_vs_time_sim_{sim_test[i]}.png')
    plt.savefig(file_path, dpi=300)
    plt.show()

    print('mse:',mse)
    print('rmse:',rmse)
    print('mae:',mae)
    print("accuracy: ",accuracy)
    print(f" Vout vs Time plot for Simulation ID: {sim_test[i]}")
'''