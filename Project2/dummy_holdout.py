import pandas as pd

# Read the CSV file
data = pd.read_csv('consumption_and_temperatures.csv')

# Get the last 2000 rows
last_2000_cases = data.tail(200)

# Write the last 2000 cases to a new file
last_2000_cases.to_csv('holdout.csv', index=False)
