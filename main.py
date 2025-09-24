import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Setting up Metadata File
metadata_df = pd.read_csv('metadata.csv')
print(metadata_df.columns)
print(metadata_df.shape)

#Setting up Waveforms File
waveforms_df = pd.read_csv('waveforms.csv')
print(waveforms_df.columns)
print(waveforms_df.shape)

# Combining the two files into one
df = pd.merge(waveforms_df, metadata_df, on='sim_id')
print('Merged Df',df.columns)
print(df.shape)

# Dropping the columns that contain the same information
cols_drop = ['Vdd_y', 'gate_y', 'process_y', 'Cload_fF_y', 'tempC_y']
df = df.drop(columns = cols_drop, axis = 1)
new_names_dict = {'Vdd_x':'Vdd', 'tempC_x':'tempC', 'process_x':'process', 'Cload_fF_x':'Cload_fF', 'gate_x':'gate'}
df.rename(columns = new_names_dict, inplace = True)
print(df.columns)

# One hot encode the gates and the process
df = pd.get_dummies(df, columns=['gate', 'process'], drop_first=True)
print(df.head())
print(df.columns)

# print("\nMissing values per column:")
# print(df.isnull().sum())

scaler = StandardScaler()
df_to_scale = df.select_dtypes(float)
transformed_data = scaler.fit_transform(df_to_scale)
df_scaled = pd.DataFrame(transformed_data, columns = df_to_scale.columns, index = df_to_scale.index)


print(df_scaled)


# Tried to do a hier clustering to find features but it's too big and didn't want to run
# numerical_features = ['t_s', 'Vout', 'Vdd', 'tempC', 'Cload_fF', 'VinA', 'VinB',
#        'Rn_ohm', 'Rp_ohm', 'tau_f_s', 'tau_r_s', 'dt_s']

# sns.clustermap(df_scaled[numerical_features], method='average', metric='euclidean', figsize=(15,40))
# plt.show()