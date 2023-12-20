import pandas as pd

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# dir_name = './'
dir_name = '/kaggle/input/llm-detect-ai-generated-text/'

train_data = pd.read_csv(dir_name + 'train_essays.csv')
test_data = pd.read_csv(dir_name + 'test_essays.csv')
train_prompts = pd.read_csv(dir_name + 'train_prompts.csv')

print(train_data.head())
print(test_data.head())
print(train_prompts.head())

ids = test_data['id'].tolist()

n = len(ids)

generated = [0.5] * n

output = pd.DataFrame({'id': ids, 'generated': generated})
output.to_csv('submission.csv', index=False)
