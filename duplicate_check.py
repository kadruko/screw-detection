import pandas as pd

train_data = pd.read_csv('train.csv')
valid_data = pd.read_csv('valid.csv')
test_data = pd.read_csv('test.csv')

# Merge the dataframes
data = pd.concat([train_data, valid_data, test_data])

# Reindex the data
data.reset_index(inplace=True, drop=True)

# Check for duplicates
duplicates = data.duplicated()

# Print the number of duplicates
print(f"Number of duplicates: {duplicates.sum()}")