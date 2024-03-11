import pandas as pd
from torch import tensor

# Load the CSV file
df = pd.read_csv('./Project2/consumption_and_temperatures.csv')




# Select the first three columns
No1 = df.iloc[:, :3]
print(No1[:50])

# Create a new formatted dataframe
new_df = pd.DataFrame()
new_df['timestamp'] = No1.iloc[24:, 0].reset_index(drop=True)
for col in range(24):
    new_df[f'time_{col+1}'] = No1.iloc[col:-24+col, 1].reset_index(drop=True)
    new_df[f'temp_{col+1}'] = No1.iloc[col:-24+col, 2].reset_index(drop=True)
new_df['temp_now'] = No1.iloc[24:, 2].reset_index(drop=True)
new_df['target'] = No1.iloc[24:, 1].reset_index(drop=True)

# Display the new dataframe
print(new_df)
from torch.utils.data import Dataset, DataLoader
import torch

torch_Data = {}
for column in new_df.columns:
    torch_Data[column] = torch.tensor(new_df[column].values)

train_dataloader = DataLoader(torch_Data, batch_size=32, shuffle=False)