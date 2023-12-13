import pandas
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def read_training_data() -> zip:
    df = pd.read_csv('./train.csv')
    # print(df.head())

    labels = df['label'].values
    pixels = df.drop('label', axis=1).values
    pixels = pixels / 255.0

    training_inputs = [np.reshape(x, (784, 1)) for x in pixels]
    training_results = [vectorized_result(y) for y in labels]

    training_data = zip(training_inputs, training_results)
    return training_data


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
    training_data = read_training_data()
    test_input = read_test_input()

    # print(training_data)
    # print(test_input[0])
    pass


if __name__ == '__main__':
    main()
