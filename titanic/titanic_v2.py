import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import os
import torch
import torch.nn as nn
import torch.optim as optim


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(6, 30)
        self.fc2 = nn.Linear(30, 1)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = torch.tanh(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def train(X, y):
    model = Net()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    X = torch.tensor(X.values, dtype=torch.float32)
    y = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)

    epochs = 1000

    for epoch in range(epochs):

        optimizer.zero_grad()

        outputs = model(X)

        loss = criterion(outputs, y)

        loss.backward()

        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}')


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

    X_test = pd.get_dummies(test_data[features])

    model = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=10)
    model.fit(X, y)

    predictions1 = model.predict(X)

    m = len(y)

    count = 0

    for i in range(m):
        if y[i] == predictions1[i]:
            count += 1

    print(f'rate={count / m}')

    predictions = model.predict(X_test)

    output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
    output.to_csv('submission.csv', index=False)

    print('submitted')


if __name__ == '__main__':
    main()
