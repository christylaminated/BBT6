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