import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

train_data = pd.read_csv('./train.csv')

print(train_data.head())

print('Number of Training Examples = {}'.format(train_data.shape[0]))

print(train_data.info())


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        hidden_unit = 10
        self.fc1 = nn.Linear(8, hidden_unit)
        self.fc2 = nn.Linear(hidden_unit, 1)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = torch.tanh(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


def train(model: Net, optimizer: optim.Adam, train_loader: DataLoader):
    model.train()

    criterion = nn.MSELoss()

    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()

        outputs = model(data)

        loss = criterion(outputs, target)

        loss.backward()

        optimizer.step()


def test(model: Net, test_loader: DataLoader):
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)

            output_tensor = torch.where(output >= 0.5, torch.tensor(1.0), torch.tensor(0.0))

            correct += output_tensor.eq(target.view_as(output_tensor)).sum().item()
            total += len(data)

    print(f'test correct rate: {correct / total}')


gb = train_data.groupby('Status')
print(gb)

print(gb.size())

print(gb.mean())


def map_bool_values(value):
    return 1 if value == 'Y' else 0


def preprocess_data(data: pd.DataFrame):
    status_mapping = {'C': 0, 'CL': 1, '1': 2}
    data['Status'] = data['Status'].map(status_mapping)

    sex_mapping = {'M': 0, 'F': 1}
    data['Sex'] = data['Sex'].map(sex_mapping)

    bool_items = ['Ascites', 'Hepatomegaly', 'Spiders', 'Edema']

    for column in bool_items:
        data[column] = data[column].map(map_bool_values)

    return data


train_data = preprocess_data(train_data)

y = train_data['Status']

features = ['N_Days', 'Age', 'Sex', 'Ascites', 'Hepatomegaly', 'Spiders', 'Edema',
            'Bilirubin', 'Cholesterol', 'Albumin', 'Copper', 'Alk_Phos', 'SGOT',
            'Tryglicerides', 'Platelets', 'Prothrombin', 'Stage']

X = pd.get_dummies(train_data[features])

model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=1)
model.fit(X, y)

predictions1 = model.predict(X)

m = len(y)

count = sum(1 for i in range(m) if y[i] == predictions1[i])

print(f'rate={count / m}')

X = torch.tensor(X.values, dtype=torch.float32)
y = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.02, random_state=42)

batch_size = 30
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

model = Net()

epochs = 100

optimizer = optim.Adam(model.parameters(), lr=0.001)

test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

for i in range(epochs):
    train(model, optimizer, train_loader)
    test(model, test_loader)

test_data = pd.read_csv('./test.csv')
X_test = pd.get_dummies(test_data[features])
predictions = model.predict(X_test)


def create_column(predictions, label):
    return [(2 / 3 if pred == label else 1 / 3) for pred in predictions]


output = pd.DataFrame({
    'id': test_data['id'],
    'Status_C': create_column(predictions, 'C'),
    'Status_CL': create_column(predictions, 'CL'),
    'Status_D': create_column(predictions, 'D'),
})

output.to_csv('submission.csv', index=False)
