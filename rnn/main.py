import numpy as np
from collections import defaultdict
from torch.utils import data

np.random.seed(42)


def generate_dataset(num_sequences=100):
    """
    Generates a number of sequences as our dataset.

    Args:
     `num_sequences`: the number of sequences to be generated.

    Returns a list of sequences.
    """
    samples = []

    for _ in range(num_sequences):
        num_tokens = np.random.randint(1, 10)
        sample = ['a'] * num_tokens + ['b'] * num_tokens + ['EOS']
        samples.append(sample)

    return samples


sequences = generate_dataset()

print('A single sample from the generated dataset:')
print(sequences[0])


def sequences_to_dicts(sequences):
    """
    Creates word_to_idx and idx_to_word dictionaries for a list of sequences.
    """

    flatten = lambda l: [item for sublist in l for item in sublist]

    all_words = flatten(sequences)

    word_count = defaultdict(int)
    for word in flatten(sequences):
        word_count[word] += 1

    word_count = sorted(list(word_count.items()), key=lambda l: -l[1])

    unique_words = [item[0] for item in word_count]

    unique_words.append('UNK')

    num_sentences, vocab_size = len(sequences), len(unique_words)

    word_to_idx = defaultdict(lambda: num_words)
    idx_to_word = defaultdict(lambda: 'UNK')

    for idx, word in enumerate(unique_words):
        word_to_idx[word] = idx
        idx_to_word[idx] = word

    return word_to_idx, idx_to_word, num_sentences, vocab_size


word_to_idx, idx_to_word, num_sequences, vocab_size = sequences_to_dicts(sequences)

print(f'We have {num_sequences} sentences and {len(word_to_idx)} unique tokens in our dataset (including UNK).\n')
print('The index of \'b\' is', word_to_idx['b'])
print(f'The word corresponding to index 1 is \'{idx_to_word[1]}\'')


class Dataset(data.Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        X = self.inputs[index]
        y = self.targets[index]

        return X, y


def create_datasets(sequences, dataset_class, p_train=0.8, p_val=0.1, p_test=0.1):
    num_train = int(len(sequences) * p_train)
    num_val = int(len(sequences) * p_val)
    num_test = int(len(sequences) * p_test)

    sequences_train = sequences[:num_train]
    sequences_val = sequences[num_train:num_train + num_val]
    sequences_test = sequences[-num_test:]

    def get_inputs_targets_from_sequences(sequences):
        inputs, targets = [], []

        for sequence in sequences:
            inputs.append(sequence[:-1])
            targets.append(sequence[1:])

        return inputs, targets

    inputs_train, targets_train = get_inputs_targets_from_sequences(sequences_train)
    inputs_val, targets_val = get_inputs_targets_from_sequences(sequences_val)
    inputs_test, targets_test = get_inputs_targets_from_sequences(sequences_test)

    training_set = dataset_class(inputs_train, targets_train)
    validation_set = dataset_class(inputs_val, targets_val)
    test_set = dataset_class(inputs_test, targets_test)

    return training_set, validation_set, test_set


training_set, validation_set, test_set = create_datasets(sequences, Dataset)

print(f'We have {len(training_set)} samples in the training set.')
print(f'We have {len(validation_set)} samples in the validation set.')
print(f'We have {len(test_set)} samples in the test set.')


def one_hot_encode(idx, vocab_size):
    """
    One-hot encodes a single word given its index and the size of the vocabulary.

    Args:
     `idx`: the index of the given word
     `vocab_size`: the size of the vocabulary

    Returns a 1-D numpy array of length `vocab_size`.
    """

    one_hot = np.zeros(vocab_size)

    one_hot[idx] = 1.0

    return one_hot


def one_hot_encode_sequence(sequence, vocab_size):
    """
    One-hot encodes a sequence of words given a fixed vocabulary size.

    Args:
     `sentence`: a list of words to encode
     `vocab_size`: the size of the vocabulary

    Returns a 3-D numpy array of shape (num words, vocab size, 1).
    """

    encoding = np.array([one_hot_encode(word_to_idx[word], vocab_size) for word in sequence])

    encoding = encoding.reshape(encoding.shape[0], encoding.shape[1], 1)

    return encoding


test_word = one_hot_encode(word_to_idx['a'], vocab_size)
print(f'Our one-hot encoding of \'a\' has shape {test_word.shape}.')

test_sentence = one_hot_encode_sequence(['a', 'b'], vocab_size)
print(f'Our one-hot encoding of \'a b\' has shape {test_sentence.shape}.')

hidden_size = 50
vocab_size = len(word_to_idx)


def init_orthogonal(param):
    """
    Initializes weight parameters orthogonally.

    Refer to this paper for an explanation of this initialization:
    https://arxiv.org/abs/1312.6120
    """
    if param.ndim < 2:
        raise ValueError("Only parameters with 2 or more dimensions are supported.")

    rows, cols = param.shape

    new_param = np.random.randn(rows, cols)

    if rows < cols:
        new_param = new_param.T

    q, r = np.linalg.qr(new_param)

    d = np.diag(r, 0)
    ph = np.sign(d)
    q *= ph

    if rows < cols:
        q = q.T

    new_param = q

    return new_param


def init_rnn(hidden_size, vocab_size):
    """
    Initializes our recurrent neural network.

    Args:
     `hidden_size`: the dimensions of the hidden state
     `vocab_size`: the dimensions of our vocabulary
    """

    U = np.zeros((hidden_size, vocab_size))

    V = np.zeros((hidden_size, hidden_size))

    W = np.zeros((vocab_size, hidden_size))

    b_hidden = np.zeros((hidden_size, 1))

    b_out = np.zeros((vocab_size, 1))

    U = init_orthogonal(U)
    V = init_orthogonal(V)
    W = init_orthogonal(W)

    return U, V, W, b_hidden, b_out


params = init_rnn(hidden_size=hidden_size, vocab_size=vocab_size)


def sigmoid(x, derivative=False):
    """
    Computes the element-wise sigmoid activation function for an array x.

    Args:
     `x`: the array where the function is applied
     `derivative`: if set to True will return the derivative instead of the forward pass
    """
    x_safe = x + 1e-12
    f = 1 / (1 + np.exp(-x_safe))

    if derivative:
        return f * (1 - f)
    else:
        return f


def tanh(x, derivative=False):
    """
    Computes the element-wise tanh activation function for an array x.

    Args:
     `x`: the array where the function is applied
     `derivative`: if set to True will return the derivative instead of the forward pass
    """
    x_safe = x + 1e-12
    f = (np.exp(x_safe) - np.exp(-x_safe)) / (np.exp(x_safe) + np.exp(-x_safe))

    if derivative:
        return 1 - f ** 2
    else:
        return f


def softmax(x, derivative=False):
    """
    Computes the softmax for an array x.

    Args:
     `x`: the array where the function is applied
     `derivative`: if set to True will return the derivative instead of the forward pass
    """
    x_safe = x + 1e-12
    f = np.exp(x_safe) / np.sum(np.exp(x_safe))

    if derivative:
        pass
    else:
        return f


def forward_pass(inputs, hidden_state, params):
    """
    Computes the forward pass of a vanilla RNN.

    Args:
     `inputs`: sequence of inputs to be processed
     `hidden_state`: an already initialized hidden state
     `params`: the parameters of the RNN
    """

    U, V, W, b_hidden, b_out = params

    outputs, hidden_states = [], []

    for t in range(len(inputs)):
        hidden_state = tanh(np.dot(U, inputs[t]) + np.dot(V, hidden_state) + b_hidden)

        out = softmax(np.dot(W, hidden_state) + b_out)

        outputs.append(out)
        hidden_states.append(hidden_state.copy())

    return outputs, hidden_states


test_input_sequence, test_target_sequence = training_set[0]

test_input = one_hot_encode_sequence(test_input_sequence, vocab_size)
test_target = one_hot_encode_sequence(test_target_sequence, vocab_size)

hidden_state = np.zeros((hidden_size, 1))

outputs, hidden_states = forward_pass(test_input, hidden_state, params)

print('Input sequence:')
print(test_input_sequence)

print('\nTarget sequence:')
print(test_target_sequence)

print('\nPredicted sequence:')
print([idx_to_word[np.argmax(output)] for output in outputs])


def clip_gradient_norm(grads, max_norm=0.25):
    """
    Clips gradients to have a maximum norm of `max_norm`.
    This is to prevent the exploding gradients problem.
    """

    max_norm = float(max_norm)
    total_norm = 0

    for grad in grads:
        grad_norm = np.sum(np.power(grad, 2))
        total_norm += grad_norm

    total_norm = np.sqrt(total_norm)

    clip_coef = max_norm / (total_norm + 1e-6)

    if clip_coef < 1:
        for grad in grads:
            grad *= clip_coef

    return grads


def backward_pass(inputs, outputs, hidden_states, targets, params):
    """
    Computes the backward pass of a vanilla RNN.

    Args:
     `inputs`: sequence of inputs to be processed
     `outputs`: sequence of outputs from the forward pass
     `hidden_states`: sequence of hidden_states from the forward pass
     `targets`: sequence of targets
     `params`: the parameters of the RNN
    """

    U, V, W, b_hidden, b_out = params

    d_U, d_V, d_W = np.zeros_like(U), np.zeros_like(V), np.zeros_like(W)
    d_b_hidden, d_b_out = np.zeros_like(b_hidden), np.zeros_like(b_out)

    d_h_next = np.zeros_like(hidden_states[0])
    loss = 0

    for t in reversed(range(len(outputs))):
        loss += -np.mean(np.log(outputs[t] + 1e-12) * targets[t])

        d_o = outputs[t].copy()
        d_o[np.argmax(targets[t])] -= 1

        d_W += np.dot(d_o, hidden_states[t].T)
        d_b_out += d_o

        d_h = np.dot(W.T, d_o) + d_h_next

        d_f = tanh(hidden_states[t], derivative=True) * d_h
        d_b_hidden += d_f

        d_U += np.dot(d_f, inputs[t].T)

        d_V += np.dot(d_f, hidden_states[t - 1].T)
        d_h_next = np.dot(V.T, d_f)

    grads = d_U, d_V, d_W, d_b_hidden, d_b_out

    grads = clip_gradient_norm(grads)

    return loss, grads


loss, grads = backward_pass(test_input, outputs, hidden_states, test_target, params)

print('We get a loss of:')
print(loss)


def update_parameters(params, grads, lr=1e-3):
    for param, grad in zip(params, grads):
        param -= lr * grad

    return params


import matplotlib.pyplot as plt

num_epochs = 1000

params = init_rnn(hidden_size=hidden_size, vocab_size=vocab_size)

hidden_state = np.zeros((hidden_size, 1))

training_loss, validation_loss = [], []

for i in range(num_epochs):

    epoch_training_loss = 0
    epoch_validation_loss = 0

    for inputs, targets in validation_set:
        inputs_one_hot = one_hot_encode_sequence(inputs, vocab_size)
        targets_one_hot = one_hot_encode_sequence(targets, vocab_size)

        hidden_state = np.zeros_like(hidden_state)

        outputs, hidden_states = forward_pass(inputs_one_hot, hidden_state, params)

        loss, _ = backward_pass(inputs_one_hot, outputs, hidden_states, targets_one_hot, params)

        epoch_validation_loss += loss

    for inputs, targets in training_set:

        inputs_one_hot = one_hot_encode_sequence(inputs, vocab_size)
        targets_one_hot = one_hot_encode_sequence(targets, vocab_size)

        hidden_state = np.zeros_like(hidden_state)

        outputs, hidden_states = forward_pass(inputs_one_hot, hidden_state, params)

        loss, grads = backward_pass(inputs_one_hot, outputs, hidden_states, targets_one_hot, params)

        if np.isnan(loss):
            raise ValueError('Gradients have vanished!')

        params = update_parameters(params, grads, lr=3e-4)

        epoch_training_loss += loss

    training_loss.append(epoch_training_loss / len(training_set))
    validation_loss.append(epoch_validation_loss / len(validation_set))

    if i % 100 == 0:
        print(f'Epoch {i}, training loss: {training_loss[-1]}, validation loss: {validation_loss[-1]}')

inputs, targets = test_set[1]

inputs_one_hot = one_hot_encode_sequence(inputs, vocab_size)
targets_one_hot = one_hot_encode_sequence(targets, vocab_size)

hidden_state = np.zeros((hidden_size, 1))

outputs, hidden_states = forward_pass(inputs_one_hot, hidden_state, params)
output_sentence = [idx_to_word[np.argmax(output)] for output in outputs]
print('Input sentence:')
print(inputs)

print('\nTarget sequence:')
print(targets)

print('\nPredicted sequence:')
print([idx_to_word[np.argmax(output)] for output in outputs])

epoch = np.arange(len(training_loss))
plt.figure()
plt.plot(epoch, training_loss, 'r', label='Training loss', )
plt.plot(epoch, validation_loss, 'b', label='Validation loss')
plt.legend()
plt.xlabel('Epoch'), plt.ylabel('NLL')
plt.show()


def freestyle(params, sentence='', num_generate=4):
    """
    Takes in a sentence as a string and outputs a sequence
    based on the predictions of the RNN.

    Args:
     `params`: the parameters of the network
     `sentence`: string with whitespace-separated tokens
     `num_generate`: the number of tokens to generate
    """
    sentence = sentence.split(' ')

    sentence_one_hot = one_hot_encode_sequence(sentence, vocab_size)

    hidden_state = np.zeros((hidden_size, 1))

    outputs, hidden_states = forward_pass(sentence_one_hot, hidden_state, params)

    output_sentence = sentence

    word = idx_to_word[np.argmax(outputs[-1])]
    output_sentence.append(word)

    for i in range(num_generate):
        output = outputs[-1]
        hidden_state = hidden_states[-1]

        output = output.reshape(1, output.shape[0], output.shape[1])

        outputs, hidden_states = forward_pass(output, hidden_state, params)

        word = idx_to_word[np.argmax(outputs)]

        output_sentence.append(word)

    return output_sentence


print('Example:')
print(freestyle(params, sentence='a a a a a b'))
