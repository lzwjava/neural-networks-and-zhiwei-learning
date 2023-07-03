import pickle
import gzip

import numpy as np

import matplotlib
import matplotlib.pyplot as plt


def load_data():
    f = gzip.open('mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(
        f, encoding='latin1')
    f.close()
    return (training_data, validation_data, test_data)


def print_shape(array):
    arr = np.array(array)
    print(arr.shape)


def load_data_wrapper():
    tr_d, va_d, te_d = load_data()

    # print(tr_d[0])
    # print_shape(tr_d[0])
    # print(tr_d[0][1])
    # print_shape(tr_d[0][1])
    # print_shape(training_inputs)

    # print(training_inputs[0])
    # print_shape(training_inputs[0])
    # print(training_inputs[0])
    print(tr_d[1])
    print_shape(tr_d[1])

    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)

    print(list(training_data)[0:1])

    print_shape(training_inputs[0])
    print_shape(training_results[0])

    print(type(training_inputs))

    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])

    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return (training_data, validation_data, test_data)


def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


td, vd, td1 = load_data_wrapper()
