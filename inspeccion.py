import pandas as pd

training = pd.read_parquet("Training.parquet")
testing = pd.read_parquet("Testing.parquet")

print("TRAINING")
print(training.shape)
print(training.info())
print(training.head())
print(training.isnull().sum())

print("\nTESTING")
print(testing.shape)
print(testing.info())
print(testing.head())
print(testing.isnull().sum())

print("\nCOLUMNAS")
print(training.columns.tolist())