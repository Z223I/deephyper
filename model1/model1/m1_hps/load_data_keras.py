"""Module to load data for Model 1."""
import os
import numpy as np
from tensorflow.keras.utils import to_categorical
import tarfile


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
    data_source = os.path.dirname(os.path.abspath(__file__))
    data_source = os.path.join(data_source, 'data')

    path = os.path.join(data_source, "data.tar.gz")



    """
Xtrain = np.loadtxt(f"{baseDirectory}aws/XTrain.at", delimiter=",")
Ytrain = np.loadtxt(f"{baseDirectory}aws/YTrain.at", delimiter=",", dtype=np.int32)
Ytrain = to_categorical(Ytrain)

print(f'Xtrain shape: {Xtrain.shape}')
print(f'Ytrain shape: {Ytrain.shape}')

countClasses = Ytrain.shape[1]
print(f'Classes: {countClasses}')
print()
print('tensorboard --logdir logs/scalars')
print('http://localhost:6006/')

Xdev = np.loadtxt(f"{baseDirectory}aws/XDev.at", delimiter=",")
Ydev = np.loadtxt(f"{baseDirectory}aws/YDev.at", delimiter=",")
Ydev = to_categorical(Ydev)
    """

    with tarfile.open(path) as tar:
        x  = np.loadtxt(tar.extractfile('dh_data/Xtrain.txt'), delimiter=",")
        y  = np.loadtxt(tar.extractfile('dh_data/yTrain.txt'), delimiter=",", dtype=np.int32)
        y  = to_categorical(y)
        #print(f'x shape: {x.shape}')
        #print(f'y shape: {y.shape}')

        assert x.shape[0] == y.shape[0]

        #Xval    = np.loadtxt(tar.extractfile('dh_data/Xdev.txt'), delimiter=",")
        #yVal    = np.loadtxt(tar.extractfile('dh_data/yDev.txt'), delimiter=",", dtype=np.int32)
        #yVal    = to_categorical(yVal)
        #print(f'Xval shape: {Xval.shape}')
        #print(f'yVal shape: {yVal.shape}')

    # print('vocab = {}'.format(vocab))
    # print('x.shape = {}'.format(x.shape))
    # print('xq.shape = {}'.format(xq.shape))
    # print('y.shape = {}'.format(y.shape))
    # print('story_maxlen, query_maxlen = {}, {}'.format(story_maxlen, query_maxlen))

    PROPORTION = 0.80
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
    config = {
        'proportion': .80           # A value between [0., 1.] indicating how to split data between
                                    # training set and validation set. `prop` corresponds to the
                                    # ratio of data in training set. `1.-prop` corresponds to the
                                    # amount of data in validation set.
    }
    load_data(config)
