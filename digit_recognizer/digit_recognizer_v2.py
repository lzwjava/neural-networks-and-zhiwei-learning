import math
import random

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def print_shape(array):
    arr = np.array(array)
    print(arr.shape)


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


def read_training_data() -> tuple:
    df = pd.read_csv('./train.csv')
    # print(df.head())

    labels = df['label'].values
    pixels = df.drop('label', axis=1).values
    pixels = pixels / 255.0

    shuffle_list = list(zip(pixels, labels))
    random.shuffle(shuffle_list)
    pixels, labels = zip(*shuffle_list)

    n = len(labels)

    inputs = [np.reshape(x, (784, 1)) for x in pixels]
    results = [vectorized_result(y) for y in labels]

    middle = int(n * 0.98)

    training_inputs = inputs[:middle]
    training_results = results[:middle]

    val_inputs = inputs[middle:]
    val_results = results[middle:]

    training_data = zip(training_inputs, training_results)
    val_data = zip(val_inputs, val_results)
    return training_data, val_data


def read_test_input() -> list:
    df = pd.read_csv('./test.csv')
    pixels = df.values
    pixels = pixels / 255.0
    test_input = [np.reshape(x, (784, 1)) for x in pixels]
    return test_input


def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


def draw(some_digit):
    some_digit_image = some_digit.reshape(28, 28)

    plt.imshow(some_digit_image, cmap=matplotlib.cm.binary,
               interpolation='nearest')
    plt.axis('off')
    plt.show()


def submit(test_output):
    test_n = len(test_output)
    images = [i + 1 for i in range(test_n)]

    output = pd.DataFrame({'ImageId': images, 'Label': test_output})
    output.to_csv('submission.csv', index=False)


def main():
    training_data, val_data = read_training_data()
    test_input = read_test_input()

    network = Network([784, 30, 10])
    network.SGD(training_data, epochs=100, mini_batch_size=10, eta=5e-2, val_data=val_data)

    test_output = network.cal(test_input)
    submit(test_output)


if __name__ == '__main__':
    main()
