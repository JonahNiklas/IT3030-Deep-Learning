import pandas as pd
from torch import tensor
from torch.utils.data import Dataset

# Load the CSV file
df = pd.read_csv('./Project2/consumption_and_temperatures.csv')

# Select the first three columns
No1 = df.iloc[:, :3]

# Create a new formatted dataframe
new_df = pd.DataFrame()
new_df['timestamp'] = No1.iloc[24:, 0].reset_index(drop=True)
for col in range(24):
    new_df[f'time_{col+1}'] = No1.iloc[col:-24+col, 1].reset_index(drop=True)
    new_df[f'temp_{col+1}'] = No1.iloc[col:-24+col, 2].reset_index(drop=True)
new_df['temp_now'] = No1.iloc[24:, 2].reset_index(drop=True)
new_df['target'] = No1.iloc[24:, 1].reset_index(drop=True)

# Create columns year, month, day, hour
new_df['year'] = pd.to_datetime(new_df['timestamp']).dt.year
new_df['month'] = pd.to_datetime(new_df['timestamp']).dt.month
new_df['day'] = pd.to_datetime(new_df['timestamp']).dt.day
new_df['hour'] = pd.to_datetime(new_df['timestamp']).dt.hour

# Reorder columns
new_df = new_df[['year', 'month', 'day', 'hour', *new_df.columns[1:-4]]]

# Write new_df to a CSV file
# new_df.to_csv('24_features_CnT.csv', index=False)

# Convert new_df, into a PyTorch dataset
class DataFrameDataset(Dataset):
    def __init__(self, dataframe, feature_columns, label_column, ):
        self.data = dataframe[feature_columns].values.astype(float)
        self.num_features = len(feature_columns)
        self.labels = dataframe[label_column].values.astype(float)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return tensor(self.data[idx]).float(), tensor(self.labels[idx]).float()
    
dataset = DataFrameDataset(new_df, new_df.columns[:-1], "target")
