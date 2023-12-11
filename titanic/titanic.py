import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import os

for dirname, _, filenames in os.walk('./'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

train_data = pd.read_csv('./train.csv')
test_data = pd.read_csv('./test.csv')

print(train_data.head())

y = train_data['Survived']
features = ['Pclass', 'Sex', 'SibSp', 'Parch']

X = pd.get_dummies(train_data[features])
print(X)

X_test = pd.get_dummies(test_data[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('submission.csv', index=False)

print('submitted')
