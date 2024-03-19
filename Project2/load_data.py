from sklearn.model_selection import train_test_split
import pandas as pd
from torch import tensor
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


def preProcessData(region=1):
    # Load the CSV file
    df = pd.read_csv('./consumption_and_temperatures.csv')

    # Filter the data for the region

    df = df[["timestamp", f"NO{region}_temperature",
             f"NO{region}_consumption"]]

    df['year'] = pd.to_datetime(df['timestamp']).dt.year
    df['month'] = pd.to_datetime(df['timestamp']).dt.month
    df['day'] = pd.to_datetime(df['timestamp']).dt.day
    df['hour'] = pd.to_datetime(df['timestamp']).dt.hour

    df["consumption_t-1"] = df["NO1_consumption"].shift(1)
    # Reorder columns
    new_df = df[['year', 'month', 'day', 'hour', 'NO1_temperature',
                'consumption_t-1', 'NO1_consumption']]

    # remove the first row
    new_df = new_df.iloc[1:]
    print(new_df.head())

    # Convert DataFrame to NumPy array
    data = new_df.to_numpy()

    # Split the target and features
    X = data[:, :-1]
    y = data[:, -1]

    # Perform train-validation-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=42, shuffle=False)

    # Standardize the features using z-scores
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    # Standardize the target using z-scores
    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_val = y_scaler.transform(y_val.reshape(-1, 1)).flatten()
    y_test = y_scaler.transform(y_test.reshape(-1, 1)).flatten()
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler, y_scaler


class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, sequence_length):
        self.X = X
        self.X = [
            self.X[i:i + sequence_length] for i in range(len(self.X) - sequence_length+1)
        ]
        self.y = y[sequence_length-1:]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return tensor(self.X[idx]).float(), tensor(self.y[idx]).float()


def getTrainingSet(reshape=False, sequence_length=24, region=1):
    X_train, X_val, X_test, y_train, y_val, y_test, scaler, y_scaler = preProcessData(
        region)
    if reshape:
        return TimeSeriesDataset(X_train, y_train.reshape(-1, 1), sequence_length)
    return TimeSeriesDataset(X_train, y_train, sequence_length)


def getValidationSet(reshape=False, sequence_length=24, region=1):
    X_train, X_val, X_test, y_train, y_val, y_test, scaler, y_scaler = preProcessData(
        region)

    return TimeSeriesDataset(X_val, y_train, sequence_length)


def getTestSet(reshape=False, sequence_length=24, region=1):
    X_train, X_val, X_test, y_train, y_val, y_test, scaler, y_scaler = preProcessData(
        region)

    return TimeSeriesDataset(X_test, y_test, sequence_length)


def getTestData(region=1):
    X_train, X_val, X_test, y_train, y_val, y_test, scaler, y_scaler = preProcessData(
        region)

    return X_test, y_test


def createTestDataFromHoldOut(filepath, region=1):
    X_train, X_val, X_test, y_train, y_val, y_test, scaler, y_scaler = preProcessData(
        region)

    df = pd.read_csv(filepath)
    df = df.iloc[:, :3]
    df['year'] = pd.to_datetime(df['timestamp']).dt.year
    df['month'] = pd.to_datetime(df['timestamp']).dt.month
    df['day'] = pd.to_datetime(df['timestamp']).dt.day
    df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
    df["consumption_t-1"] = df["NO1_consumption"].shift(1)
    new_df = df[['year', 'month', 'day', 'hour', 'NO1_temperature',
                'consumption_t-1', 'NO1_consumption']]
    new_df = new_df.iloc[1:]
    data = new_df.to_numpy()
    X = data[:, :-1]
    y = data[:, -1]
    X = scaler.transform(X)
    y = y_scaler.transform(y.reshape(-1, 1)).flatten()
    return X, y
