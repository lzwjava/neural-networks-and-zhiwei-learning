import numpy as np
from collections import defaultdict
from torch.utils import data

import matplotlib.pyplot as plt

np.random.seed(42)


def sigmoid(x, derivative=False):
    x_safe = x + 1e-12
    f = 1 / (1 + np.exp(-x_safe))

    if derivative:
        return f * (1 - f)
    else:
        return f


def tanh(x, derivative=False):
    x_safe = x + 1e-12
    f = (np.exp(x_safe) - np.exp(-x_safe)) / (np.exp(x_safe) + np.exp(-x_safe))

    if derivative:
        return 1 - f ** 2
    else:
        return f


def softmax(x, derivative=False):
    x_safe = x + 1e-12
    f = np.exp(x_safe) / np.sum(np.exp(x_safe))

    if derivative:
        pass
    else:
        return f


def generate_dataset(num_sequences=100):
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
    one_hot = np.zeros(vocab_size)

    one_hot[idx] = 1.0

    return one_hot


def one_hot_encode_sequence(sequence, vocab_size):
    encoding = np.array([one_hot_encode(word_to_idx[word], vocab_size) for word in sequence])

    encoding = encoding.reshape(encoding.shape[0], encoding.shape[1], 1)

    return encoding


test_word = one_hot_encode(word_to_idx['a'], vocab_size)
print(f'Our one-hot encoding of \'a\' has shape {test_word.shape}.')

test_sentence = one_hot_encode_sequence(['a', 'b'], vocab_size)
print(f'Our one-hot encoding of \'a b\' has shape {test_sentence.shape}.')

hidden_size = 50
vocab_size = len(word_to_idx)

z_size = hidden_size + vocab_size


def init_lstm(hidden_size, vocab_size, z_size):
    W_f = np.random.randn(hidden_size, z_size)

    b_f = np.zeros((hidden_size, 1))

    W_i = np.random.randn(hidden_size, z_size)

    b_i = np.zeros((hidden_size, 1))

    W_g = np.random.randn(hidden_size, z_size)

    b_g = np.zeros((hidden_size, 1))

    W_o = np.random.randn(hidden_size, z_size)
    b_o = np.zeros((hidden_size, 1))

    W_v = np.random.randn(vocab_size, hidden_size)
    b_v = np.zeros((vocab_size, 1))

    W_f = init_orthogonal(W_f)
    W_i = init_orthogonal(W_i)
    W_g = init_orthogonal(W_g)
    W_o = init_orthogonal(W_o)
    W_v = init_orthogonal(W_v)

    return W_f, W_i, W_g, W_o, W_v, b_f, b_i, b_g, b_o, b_v


def init_orthogonal(param):
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


params = init_lstm(hidden_size=hidden_size, vocab_size=vocab_size, z_size=z_size)


def forward(inputs, h_prev, C_prev, p):
    assert h_prev.shape == (hidden_size, 1)
    assert C_prev.shape == (hidden_size, 1)

    W_f, W_i, W_g, W_o, W_v, b_f, b_i, b_g, b_o, b_v = p

    x_s, z_s, f_s, i_s, = [], [], [], []
    g_s, C_s, o_s, h_s = [], [], [], []
    v_s, output_s = [], []

    h_s.append(h_prev)
    C_s.append(C_prev)

    for x in inputs:
        z = np.row_stack((h_prev, x))
        z_s.append(z)

        f = sigmoid(np.dot(W_f, z) + b_f)
        f_s.append(f)

        i = sigmoid(np.dot(W_i, z) + b_i)
        i_s.append(i)

        g = tanh(np.dot(W_g, z) + b_g)
        g_s.append(g)

        C_prev = f * C_prev + i * g
        C_s.append(C_prev)

        o = sigmoid(np.dot(W_o, z) + b_o)
        o_s.append(o)

        h_prev = o * tanh(C_prev)
        h_s.append(h_prev)

        v = np.dot(W_v, h_prev) + b_v
        v_s.append(v)

        output = softmax(v)
        output_s.append(output)

    return z_s, f_s, i_s, g_s, C_s, o_s, h_s, v_s, output_s


inputs, targets = test_set[1]

inputs_one_hot = one_hot_encode_sequence(inputs, vocab_size)
targets_one_hot = one_hot_encode_sequence(targets, vocab_size)

h = np.zeros((hidden_size, 1))
c = np.zeros((hidden_size, 1))

z_s, f_s, i_s, g_s, C_s, o_s, h_s, v_s, outputs = forward(inputs_one_hot, h, c, params)

output_sentence = [idx_to_word[np.argmax(output)] for output in outputs]
print('Input sentence:')
print(inputs)

print('\nTarget sequence:')
print(targets)

print('\nPredicted sequence:')
print([idx_to_word[np.argmax(output)] for output in outputs])


def clip_gradient_norm(grads, max_norm=0.25):
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


def update_parameters(params, grads, lr=1e-3):
    for param, grad in zip(params, grads):
        param -= lr * grad

    return params


def backward(z, f, i, g, C, o, h, v, outputs, targets, p=params):
    W_f, W_i, W_g, W_o, W_v, b_f, b_i, b_g, b_o, b_v = p

    W_f_d = np.zeros_like(W_f)
    b_f_d = np.zeros_like(b_f)

    W_i_d = np.zeros_like(W_i)
    b_i_d = np.zeros_like(b_i)

    W_g_d = np.zeros_like(W_g)
    b_g_d = np.zeros_like(b_g)

    W_o_d = np.zeros_like(W_o)
    b_o_d = np.zeros_like(b_o)

    W_v_d = np.zeros_like(W_v)
    b_v_d = np.zeros_like(b_v)

    dh_next = np.zeros_like(h[0])
    dC_next = np.zeros_like(C[0])

    loss = 0

    for t in reversed(range(len(outputs))):
        loss += -np.mean(np.log(outputs[t]) * targets[t])

        C_prev = C[t - 1]

        dv = np.copy(outputs[t])
        dv[np.argmax(targets[t])] -= 1

        W_v_d += np.dot(dv, h[t].T)
        b_v_d += dv

        dh = np.dot(W_v.T, dv)
        dh += dh_next
        do = dh * tanh(C[t])
        do = sigmoid(o[t], derivative=True) * do

        W_o_d += np.dot(do, z[t].T)
        b_o_d += do

        dC = np.copy(dC_next)
        dC += dh * o[t] * tanh(tanh(C[t]), derivative=True)
        dg = dC * i[t]
        dg = tanh(g[t], derivative=True) * dg

        W_g_d += np.dot(dg, z[t].T)
        b_g_d += dg

        di = dC * g[t]
        di = sigmoid(i[t], True) * di
        W_i_d += np.dot(di, z[t].T)
        b_i_d += di

        df = dC * C_prev
        df = sigmoid(f[t]) * df
        W_f_d += np.dot(df, z[t].T)
        b_f_d += df

        dz = (np.dot(W_f.T, df)
              + np.dot(W_i.T, di)
              + np.dot(W_g.T, dg)
              + np.dot(W_o.T, do))
        dh_prev = dz[:hidden_size, :]
        dC_prev = f[t] * dC

    grads = W_f_d, W_i_d, W_g_d, W_o_d, W_v_d, b_f_d, b_i_d, b_g_d, b_o_d, b_v_d

    grads = clip_gradient_norm(grads)

    return loss, grads


loss, grads = backward(z_s, f_s, i_s, g_s, C_s, o_s, h_s, v_s, outputs, targets_one_hot, params)

print('We get a loss of:')
print(loss)

num_epochs = 50

z_size = hidden_size + vocab_size
params = init_lstm(hidden_size=hidden_size, vocab_size=vocab_size, z_size=z_size)

hidden_state = np.zeros((hidden_size, 1))

training_loss, validation_loss = [], []

for i in range(num_epochs):

    epoch_training_loss = 0
    epoch_validation_loss = 0

    for inputs, targets in validation_set:
        inputs_one_hot = one_hot_encode_sequence(inputs, vocab_size)
        targets_one_hot = one_hot_encode_sequence(targets, vocab_size)

        h = np.zeros((hidden_size, 1))
        c = np.zeros((hidden_size, 1))

        z_s, f_s, i_s, g_s, C_s, o_s, h_s, v_s, outputs = forward(inputs_one_hot, h, c, params)

        loss, _ = backward(z_s, f_s, i_s, g_s, C_s, o_s, h_s, v_s, outputs, targets_one_hot, params)

        epoch_validation_loss += loss

    for inputs, targets in training_set:
        inputs_one_hot = one_hot_encode_sequence(inputs, vocab_size)
        targets_one_hot = one_hot_encode_sequence(targets, vocab_size)

        h = np.zeros((hidden_size, 1))
        c = np.zeros((hidden_size, 1))

        z_s, f_s, i_s, g_s, C_s, o_s, h_s, v_s, outputs = forward(inputs_one_hot, h, c, params)

        loss, grads = backward(z_s, f_s, i_s, g_s, C_s, o_s, h_s, v_s, outputs, targets_one_hot, params)

        params = update_parameters(params, grads, lr=1e-1)

        epoch_training_loss += loss

    training_loss.append(epoch_training_loss / len(training_set))
    validation_loss.append(epoch_validation_loss / len(validation_set))

    if i % 5 == 0:
        print(f'Epoch {i}, training loss: {training_loss[-1]}, validation loss: {validation_loss[-1]}')

inputs, targets = test_set[1]

inputs_one_hot = one_hot_encode_sequence(inputs, vocab_size)
targets_one_hot = one_hot_encode_sequence(targets, vocab_size)

h = np.zeros((hidden_size, 1))
c = np.zeros((hidden_size, 1))

z_s, f_s, i_s, g_s, C_s, o_s, h_s, v_s, outputs = forward(inputs_one_hot, h, c, params)

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
