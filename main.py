import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Loading the datasets
metadata_df = pd.read_csv('metadata.csv')
waveforms_df = pd.read_csv('waveforms.csv')

df = pd.merge(waveforms_df, metadata_df, on='sim_id')

# Cleaning up data column Names
cols_drop = ['Vdd_y', 'gate_y', 'process_y', 'Cload_fF_y', 'tempC_y']
df = df.drop(columns=cols_drop, axis=1)

# Renaming the columns
new_names_dict = {
    'Vdd_x': 'Vdd',
    'tempC_x': 'tempC',
    'process_x': 'process',
    'Cload_fF_x': 'Cload_fF',
    'gate_x': 'gate'
}
df.rename(columns=new_names_dict, inplace=True)

# Missing Values in VinB to 0
df['VinB'].fillna(0, inplace=True)


# Making Voltage Derivative Column 
"""
    Can be used as a comparision point between the given result and the 
    results our model will give us. 
"""
df['dVout_dt'] = df.groupby('sim_id')['Vout'].diff() / df.groupby('sim_id')['t_s'].diff()
df['dVout_dt'].fillna(0, inplace=True) 
# In case the divisior had a 0
df.replace([np.inf, -np.inf], 0, inplace=True)

# Making Tau column 
"""
    The reason why this is being implemented is because based on t = R*C, we
    can get a rough estimate in how long it takes for the circuit to charge
    /discharge
"""
df['tau_f_est'] = df['Rn_ohm'] * df['Cload_fF']
df['tau_r_est'] = df['Rp_ohm'] * df['Cload_fF']


# Making a heatmap
num_feat = df.select_dtypes(include=np.number).drop(columns='sim_id')
corr_matrix = num_feat.corr()

plt.figure(figsize=(16, 12))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Matrix of Numerical Features')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('correlation_heatmap.png')


print("Shape of the merged dataframe before downsampling:", df.shape)

"""
    Purpose: 
    Downsamples the dataframe by comparing the Vout values.
    If the value of Vout is not above the threshold then it is not 
    keep in the dataFrame. 

    Returns:
    DataFrame
"""
def ds_Vout(group, thres=0.01):
    if group.empty:
        return pd.DataFrame()
    
    last_kept_row = group.iloc[0]
    kept_rows = [last_kept_row]

    for i in range(1, len(group)):
        curr_row = group.iloc[i]
        if abs(curr_row['Vout'] - last_kept_row['Vout']) > thres:
            kept_rows.append(curr_row)
            last_kept_row = curr_row
    return pd.DataFrame(kept_rows)

# Downsampling the data
downsampled_df = df.groupby('sim_id').apply(ds_Vout).reset_index(drop=True)
#print("Shape of the dataframe after downsampling:", downsampled_df.shape)

# One-hot encode gate and process features
processed_df = pd.get_dummies(downsampled_df, columns=['gate', 'process'], 
                              drop_first=True)

# Scale numerical features
numerical_cols = processed_df.select_dtypes(include=np.number).columns
# We don't want to scale sim_id or the one-hot encoded columns
cols_scale = [col for col in numerical_cols if col not in ['sim_id'] and 
              not col.startswith('gate_') and not col.startswith('process_')]

scaler = StandardScaler()
processed_df[cols_scale] = scaler.fit_transform(processed_df[cols_scale])

print("\nFirst 5 rows of the processed dataframe:")
print(processed_df.head())

# Saving to a new csv
processed_df.to_csv('processed_waveforms.csv', index=False)