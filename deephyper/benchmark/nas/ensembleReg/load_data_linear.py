import os
import numpy as np
from deephyper.benchmark.benchmark_functions_wrappers import linear_

HERE = os.path.dirname(os.path.abspath(__file__))

def load_data(dim=10):
    """
    Generate data for linear function -sum(x_i).

    Return:
        Tuple of Numpy arrays: ``(train_X, train_y), (valid_X, valid_y)``.
    """
    rs = np.random.RandomState(2018)
    size = 100000
    prop = 0.80
    f, (a, b), _ = linear_()
    d = b - a
    x = np.array([a + rs.random(dim) * d for i in range(size)])
    y = np.array([[f(v)] for v in x])

    sep_index = int(prop * size)
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
