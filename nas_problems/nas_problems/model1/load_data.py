"""Module to load data for Model 1."""
import os
import numpy as np
#from tensorflow.keras.utils import to_categorical
import tarfile

#
#
# Note:  This is a copy of model1/model1/m1_hps/load_data_pytorch.py.
# It is a copy because I didn't want to break the other code.
#
#

def load_data(config):
    """
    Load data for Model 1.

    Args:
        config (dict): Configuration dictionary.

    Returns:
        tuple(tuple(ndarray, ndarray), tuple(ndarray, ndarray)): of Numpy arrays: `(train_X, train_y), (valid_X, valid_y)`.
    """
    data_source = os.path.dirname(os.path.abspath(__file__))
    data_source = os.path.join(data_source, 'data')

    #tarFilename    = "data.tar.gz"
    #XTrainFilename = 'dh_data/Xtrain.txt'
    #YTrainFilename = 'dh_data/yTrain.txt'

    tarFilename    = "data485.tar.gz"
    XTrainFilename = 'XTrain.txt'
    YTrainFilename = 'YTrain.txt'

    tarFilename    = "data.tar.gz"
    XTrainFilename = 'dh_data/Xtrain.txt'
    YTrainFilename = 'dh_data/yTrain.txt'

    path = os.path.join(data_source, tarFilename)

    # Extract data from tar file.
    with tarfile.open(path) as tar:
        x  = np.loadtxt(tar.extractfile(XTrainFilename), delimiter=",")
        y  = np.loadtxt(tar.extractfile(YTrainFilename), delimiter=",", dtype=np.int32)
        # floor and int are a replacement for keras to_categorical
        y  = (np.floor(y)).astype(int)
        #print(f'x shape: {x.shape}')
        #print(f'y shape: {y.shape}')

        assert x.shape[0] == y.shape[0]

    #yHead = y[:10]
    #print(f"yHead: {yHead}")

    PROPORTION = config['proportion']
    size = x.shape[0]

    sep_index = int(PROPORTION * size)
    train_X = x[:sep_index]
    train_y = y[:sep_index]

    valid_X = x[sep_index:]
    valid_y = y[sep_index:]

    key = 'print_shape'
    if key in config.keys() and config['print_shape'] == 1:
        print(f'train_X shape: {np.shape(train_X)}')
        print(f'train_y shape: {np.shape(train_y)}')
        print(f'valid_X shape: {np.shape(valid_X)}')
        print(f'valid_y shape: {np.shape(valid_y)}')

    return (train_X, train_y), (valid_X, valid_y)

if __name__ == '__main__':
    config = {
        'proportion': .90,          # A value between [0., 1.] indicating how to split data between
                                    # training set and validation set. `prop` corresponds to the
                                    # ratio of data in training set. `1.-prop` corresponds to the
                                    # amount of data in validation set.
        'print_shape': 0            # Print the data shape.
    }
    (train_X, train_y), (valid_X, valid_y) = load_data(config)

    print(f'train_X shape: {np.shape(train_X)}')
    print(f'train_y shape: {np.shape(train_y)}')
    print(f'valid_X shape: {np.shape(valid_X)}')
    print(f'valid_y shape: {np.shape(valid_y)}')
