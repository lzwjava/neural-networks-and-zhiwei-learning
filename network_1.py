import random

import numpy as np
import matplotlib
import matplotlib.pyplot as plt


class Network(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes

        print(sizes)
        print(sizes[1:])

        # print(np.random.randn(30, 1))

        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        print('biases:', self.biases)

        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

        # print(self.weights)
        print('weights:', self.weights)
        print('shape:')
        print_shape(self.weights[0])

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        training_data = list(training_data)
        n = len(training_data)

        print(mini_batch_size)
        print(epochs)

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)

        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {}: {}/{}".format(j, self.evalute(test_data), n_test))
            else:
                print("Epoch {} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        pass

    def evalute(self, test_data):
        print('test_data', test_data[0])
        # draw(test_data[0][0])
        print('shape of test_data[0][0]:')
        print_shape(test_data[0][0])
        
        print(type(test_data))
        # print_shape(test_data[1])

        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def feedforward(self, a):
        # i = 0
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
            # if (i < 5):
            #     print('b:', b)
            #     print('w:', w)
            # i += 1
        return a


def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))


def draw(some_digit):
    some_digit_image = some_digit.reshape(28, 28)

    plt.imshow(some_digit_image, cmap=matplotlib.cm.binary,
               interpolation='nearest')
    plt.axis('off')
    plt.show()


def print_shape(array):
    arr = np.array(array)
    print(arr.shape)
