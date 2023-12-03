import h5py
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.framework.ops import EagerTensor
from tensorflow.python.ops.resource_variable_ops import ResourceVariable
import time

train_dataset = h5py.File('datasets/train_signs.h5', "r")
test_dataset = h5py.File('datasets/test_signs.h5', "r")

x_train = tf.data.Dataset.from_tensor_slices(train_dataset['train_set_x'])
y_train = tf.data.Dataset.from_tensor_slices(train_dataset['train_set_y'])

x_test = tf.data.Dataset.from_tensor_slices(test_dataset['test_set_x'])
y_test = tf.data.Dataset.from_tensor_slices(test_dataset['test_set_y'])

print(type(x_train))

print(x_train.element_spec)

print(next(iter(x_train)))

unique_labels = set()
for element in y_train:
    unique_labels.add(element.numpy())
print(unique_labels)

images_iter = iter(x_train)
labels_iter = iter(y_train)
plt.figure(figsize=(10, 10))
for i in range(25):
    ax = plt.subplot(5, 5, i + 1)
    plt.imshow(next(images_iter).numpy().astype("uint8"))
    plt.title(next(labels_iter).numpy().astype("uint8"))
    plt.axis("off")


def normalize(image):
    """
    Transform an image into a tensor of shape (64 * 64 * 3, )
    and normalize its components.

    Arguments
    image - Tensor.

    Returns:
    result -- Transformed tensor
    """
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.reshape(image, [-1, ])
    return image


new_train = x_train.map(normalize)
new_test = x_test.map(normalize)

new_train.element_spec

print(next(iter(new_train)))


# GRADED FUNCTION: linear_function

def linear_function():
    """
    Implements a linear function:
            Initializes X to be a random tensor of shape (3,1)
            Initializes W to be a random tensor of shape (4,3)
            Initializes b to be a random tensor of shape (4,1)
    Returns:
    result -- Y = WX + b
    """

    np.random.seed(1)

    """
    Note, to ensure that the "random" numbers generated match the expected results,
    please create the variables in the order given in the starting code below.
    (Do not re-arrange the order).
    """
    # (approx. 4 lines)
    X = tf.constant(np.random.randn(3, 1), name="X")
    W = tf.constant(np.random.randn(4, 3), name="W")
    b = tf.constant(np.random.randn(4, 1), name="b")
    Y = tf.add(tf.matmul(W, X), b)
    return Y


result = linear_function()
print(result)

assert type(result) == EagerTensor, "Use the TensorFlow API"
assert np.allclose(result, [[-2.15657382], [2.95891446], [-1.08926781], [-0.84538042]]), "Error"
print("\033[92mAll test passed")


# GRADED FUNCTION: sigmoid

def sigmoid(z):
    """
    Computes the sigmoid of z

    Arguments:
    z -- input value, scalar or vector

    Returns:
    a -- (tf.float32) the sigmoid of z
    """
    # tf.keras.activations.sigmoid requires float16, float32, float64, complex64, or complex128.

    # (approx. 2 lines)
    # z = ...
    # a = ...
    # YOUR CODE STARTS HERE

    z = tf.cast(z, tf.float32)
    a = tf.keras.activations.sigmoid(z)

    # YOUR CODE ENDS HERE
    return a


result = sigmoid(-1)
print("type: " + str(type(result)))
print("dtype: " + str(result.dtype))
print("sigmoid(-1) = " + str(result))
print("sigmoid(0) = " + str(sigmoid(0.0)))
print("sigmoid(12) = " + str(sigmoid(12)))


def sigmoid_test(target):
    result = target(0)
    assert (type(result) == EagerTensor)
    assert (result.dtype == tf.float32)
    assert sigmoid(0) == 0.5, "Error"
    assert sigmoid(-1) == 0.26894143, "Error"
    assert sigmoid(12) == 0.99999386, "Error"

    print("\033[92mAll test passed")


sigmoid_test(sigmoid)


# GRADED FUNCTION: one_hot_matrix
def one_hot_matrix(label, C=6):
    # (approx. 1 line)
    # one_hot = None(None(None, None, None), shape=[C, ])
    # YOUR CODE STARTS HERE

    one_hot = None

    # YOUR CODE ENDS HERE
    return one_hot


def one_hot_matrix_test(target):
    label = tf.constant(1)
    C = 4
    result = target(label, C)
    print("Test 1:", result)
    assert result.shape[0] == C, "Use the parameter C"
    assert np.allclose(result, [0., 1., 0., 0.]), "Wrong output. Use tf.one_hot"
    label_2 = [2]
    result = target(label_2, C)
    print("Test 2:", result)
    assert result.shape[0] == C, "Use the parameter C"
    assert np.allclose(result, [0., 0., 1., 0.]), "Wrong output. Use tf.reshape as instructed"

    print("\033[92mAll test passed")


one_hot_matrix_test(one_hot_matrix)

new_y_test = y_test.map(one_hot_matrix)
new_y_train = y_train.map(one_hot_matrix)

print(next(iter(new_y_test)))
