"""Module to load data for Model 1."""
import os
import numpy as np
from tensorflow.keras.utils import to_categorical
import tarfile

def get_data(f):
    # sourcery skip: inline-immediately-returned-variable
    """Given a file-like object, read the file."""
    data = np.load(f)
    return data

def load_data(config):
    """
    Generate a random distribution of data for polynome_2 function.

    Generate a random distribution of data for polynome_2 function: -SUM(X**2) where
    "**" is an element wise operator in the continuous range [a, b].

    Args:
        config (dict): Configuration dictionary.

    Returns:
        tuple(tuple(ndarray, ndarray), tuple(ndarray, ndarray)): of Numpy arrays: `(train_X, train_y), (valid_X, valid_y)`.
    """
    if config['data_source']:
        data_source = config['data_source']
    else:
        data_source = os.path.dirname(os.path.abspath(__file__))
        data_source = os.path.join(data_source, 'data')

    path = os.path.join(data_source, "data.tar.gz")

    with tarfile.open(path) as tar:
        x  = get_data(tar.extractfile('Xtrain.txt'))
        y  = get_data(tar.extractfile('yTrain.txt'))
        y  = to_categorical(y)
        print(f'x shape: {x.shape}')
        print(f'y shape: {y.shape}')

        assert x.shape == y.shape

        Xval    = get_data(tar.extractfile('Xdev.txt'))
        yVal    = get_data(tar.extractfile('yDev.txt'))
        yVal    = to_categorical(yVal)
        print(f'Xval shape: {Xval.shape}')
        print(f'yVal shape: {yVal.shape}')

        Xtest   = get_data(tar.extractfile('Xtest.txt'))
        yTest   = get_data(tar.extractfile('yTest.txt'))
        yTest   = to_categorical(yTest)
        print(f'Xtest shape: {Xtest.shape}')
        print(f'yTest shape: {yTest.shape}')

    # print('vocab = {}'.format(vocab))
    # print('x.shape = {}'.format(x.shape))
    # print('xq.shape = {}'.format(xq.shape))
    # print('y.shape = {}'.format(y.shape))
    # print('story_maxlen, query_maxlen = {}, {}'.format(story_maxlen, query_maxlen))

    PROPORTION = config['proportion']
    size = x.shape[0]

    sep_index = int(PROPORTION * size)
    train_X = x[:sep_index]
    train_y = y[:sep_index]

    valid_X = x[sep_index:]
    valid_y = y[sep_index:]

    print(f'train_X shape: {np.shape(train_X)}')
    print(f'train_y shape: {np.shape(train_y)}')
    print(f'valid_X shape: {np.shape(valid_X)}')
    print(f'valid_y shape: {np.shape(valid_y)}')
    return (train_X, train_y), (valid_X, valid_y)

if __name__ == '__main__':
    load_data()