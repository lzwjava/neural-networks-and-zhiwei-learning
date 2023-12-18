import pandas as pd

train_data = pd.read_csv('train_essays.csv')
test_data = pd.read_csv('test_essays.csv')
train_prompts = pd.read_csv('train_prompts.csv')

print(train_data.head())
