import pandas as pd
from torch import tensor
from torch.utils.data import Dataset

# Load the CSV file
df = pd.read_csv('./Project2/consumption_and_temperatures.csv')

# Select the first three columns
df = df.iloc[:, :3]
df['year'] = pd.to_datetime(df['timestamp']).dt.year
df['month'] = pd.to_datetime(df['timestamp']).dt.month
df['day'] = pd.to_datetime(df['timestamp']).dt.day
df['hour'] = pd.to_datetime(df['timestamp']).dt.hour

df["consumption_t-1"] = df["NO1_consumption"].shift(1)

# Reorder columns
new_df = df[['year', 'month', 'day', 'hour', 'NO1_temperature',
             'consumption_t-1', 'NO1_consumption']]

# Convert new_df, into a PyTorch dataset


class DataFrameDataset(Dataset):
    def __init__(self, dataframe, feature_columns, label_column, sequence_length):
        self.data = dataframe[feature_columns].values.astype(float)
        self.num_features = len(feature_columns)
        # convert data to sequences
        self.data = [
            self.data[i: i + sequence_length] for i in range(len(self.data) - sequence_length)
        ]
        self.labels = dataframe[label_column].values.astype(float)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return tensor(self.data[idx]).float(), tensor(self.labels[idx]).float()


featues = new_df.columns[:-1]
target = new_df.columns[-1]
sequence_length = 24

dataset = DataFrameDataset(new_df, featues, target, sequence_length)
