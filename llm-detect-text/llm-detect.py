import pandas as pd

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

train_data = pd.read_csv('train_essays.csv')
test_data = pd.read_csv('test_essays.csv')
train_prompts = pd.read_csv('train_prompts.csv')

print(train_data.head())
print(test_data.head())
print(train_prompts.head())
