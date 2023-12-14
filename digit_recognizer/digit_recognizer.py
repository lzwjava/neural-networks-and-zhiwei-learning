import math

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class Network(object):

    def __init__(self, sizes):
        self.layers = len(sizes)
        self.sizes = sizes
        self.weights = [np.random.rand(sizes[i + 1], sizes[i]) for i in range(self.layers - 1)]
        self.biases = [np.random.rand(sizes[i + 1]) for i in range(self.layers - 1)]
        print('hi')

    def SGD(self, training_data: zip, epochs: int, mini_batch_size: int, eta: float,
            val_data: zip):

        training_data = list(training_data)

        n = len(training_data)

        num_batches = math.ceil(n / mini_batch_size)

        for j in range(epochs):
            for k in range(num_batches):
                start_idx = k * mini_batch_size
                end_idx = (k + 1) * mini_batch_size

                mini_batches = training_data[start_idx:end_idx]
                for mini_batch in mini_batches:
                    self.update_mini_batch(mini_batch, eta)

    def update_mini_batch(self, mini_batch, eta):
        for (x, y) in mini_batch:
            self.backprop(x, y)

    def backprop(self, x, y):
        pass

    def cost_derivative(self, output_activations, y):
        pass

    def evalute(self, test_data):
        pass

    def feedforward(self, a):
        for w, b in zip(self.weights, self.biases):
            a = sigmoid(np.dot(w, a) + b)
        return a


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

    n = len(labels)

    inputs = [np.reshape(x, (784, 1)) for x in pixels]
    results = [vectorized_result(y) for y in labels]

    middle = int(n * 0.9)

    training_inputs = inputs[:middle]
    training_results = results[:middle]

    val_inputs = inputs[middle:]
    val_results = inputs[middle:]

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


def main():
    training_data, val_data = read_training_data()
    test_input = read_test_input()

    network = Network([784, 30, 10])
    network.SGD(training_data, epochs=100, mini_batch_size=10, eta=1e-5, val_data=val_data)

    # print(training_data)
    # print(test_input[0])


if __name__ == '__main__':
    main()
