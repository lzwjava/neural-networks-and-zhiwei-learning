import random

import numpy as np


class Network(object):
    
    def print_shape(self, array):
        arr = np.array(array)
        print(arr.shape)

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        
        print(sizes)
        print(sizes[1:])
        
        # print(np.random.randn(30, 1))
        
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        print(self.biases)
        
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
        
        # print(self.weights)
        self.print_shape(self.weights[1])

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
