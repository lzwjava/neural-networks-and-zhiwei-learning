import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from pandas import DataFrame


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(7, 30)
        self.fc2 = nn.Linear(30, 1)
        self.dropout = nn.Dropout(0.25)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = torch.tanh(x)
        x = self.fc1(x)
        x = self.dropout(x)
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

        if (batch_idx + 1) % 10 == 0:
            print(f'batch {batch_idx + 1}, Loss: {loss.item()}')


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


def cal(test_data: DataFrame, features, model: Net):
    X_test_2 = pd.get_dummies(test_data[features])

    X_test_vec = torch.tensor(X_test_2.values, dtype=torch.float32)

    pred = model(X_test_vec)

    pred = pred.view(-1)

    pred = pred.detach().numpy()

    new_pred = []

    for i in range(len(pred)):
        v = 1 if pred[i] >= 0.5 else 0
        new_pred.append(v)

    output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': new_pred})
    output.to_csv('submission.csv', index=False)

    print('submitted')


def main():
    train_data = pd.read_csv('./train.csv')
    test_data = pd.read_csv('./test.csv')

    print(train_data.head())

    y = train_data['Survived']
    features = ['Age', 'Pclass', 'Sex', 'SibSp', 'Parch', 'Fare']

    train_data['Age'].fillna(0, inplace=True)
    test_data['Age'].fillna(0, inplace=True)

    train_data['Fare'].fillna(0, inplace=True)
    test_data['Fare'].fillna(0, inplace=True)

    X = pd.get_dummies(train_data[features])

    print(X)

    X = torch.tensor(X.values, dtype=torch.float32)
    y = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

    batch_size = 32
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = Net()

    epochs = 100

    optimizer = optim.Adam(model.parameters(), lr=0.1)

    for i in range(epochs):
        train(model, optimizer, train_loader)

    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    test(model, test_loader)

    cal(test_data, features, model)


if __name__ == '__main__':
    main()
