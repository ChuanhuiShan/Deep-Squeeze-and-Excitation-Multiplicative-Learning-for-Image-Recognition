from scipy.io import loadmat
import numpy as np

def load(file):
    data_set = loadmat(file)
    samples = np.transpose(data_set["X"], (3, 0, 1, 2)).astype(np.float32)
    # print(samples.shape)
    labels = np.array([x[0] for x in data_set["y"]])
    one_hot_labels = []
    for num in labels:
        one_hot = [0.] * 10
        one_hot[num % 10] = 1.
        one_hot_labels.append(one_hot)
    labels = np.array(one_hot_labels).astype(np.float32)
    # samples = np.add.reduce(samples, keepdims=True, axis=3) / 3.0
    samples = samples / 128.0 - 1
    return samples, labels

def data_iterator(samples, labels, batch_size, iteration_steps=None):
    if len(samples) != len(labels):
        raise Exception('Length of samples and labels must equal')
    if len(samples) < batch_size:
        raise Exception('Length of samples must be smaller than batch size.')
    start = 0
    step = 0
    if iteration_steps is None:
        while start < len(samples):
            end = start + batch_size
            if end < len(samples):
                yield step, samples[start:end], labels[start:end]
                step += 1
            start = end
    else:
        while step < iteration_steps:
            start = (step * batch_size) % (len(labels) - batch_size)
            yield step, samples[start:start + batch_size], labels[start:start + batch_size]
            step += 1

def random_data(samples, labels, batch_size):
    if len(samples) != len(labels):
        raise Exception('Length of samples and labels must equal')
    if len(samples) < batch_size:
        raise Exception('Length of samples must be smaller than batch size.')
    start = np.random.randint(len(samples) - batch_size)
    return samples[start:start + batch_size], labels[start:start + batch_size]
