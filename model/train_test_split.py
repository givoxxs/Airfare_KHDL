from sklearn.model_selection import train_test_split
import pandas as pd
df = pd.read_csv('../data/air_fare_raw.csv')
trainset, testset = train_test_split(df, test_size=0.2, random_state=42)
trainset.to_csv('../data/train.csv')
trainset.to_csv('../data/test.csv')
trainset.reset_index(inplace=True, drop=True)
testset.reset_index(inplace=True, drop=True)