import argparse

def add_args(parser: argparse.ArgumentParser):
    """Add common arguments that are used for every type of run."""
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate for training")
    parser.add_argument('--momentum', type=float, default=0.0, help="Momentum value for training")
    parser.add_argument('--weight-decay', type=float, default=1e-4, help="Weight decay for training")
    parser.add_argument('-e', '--num-epochs', type=int, default=1)

    parser.add_argument('--num-features', type=int, default=1690)   # Model 1.
    parser.add_argument('--num-classes', type=int, default=2)       # Model 1.
    parser.add_argument('--proportion', type=float, default=.80)    # Model 1.
                                    # A value between [0., 1.] indicating how to split data between
                                    # training set and validation set. `prop` corresponds to the
                                    # ratio of data in training set. `1.-prop` corresponds to the
                                    # amount of data in validation set.
    parser.add_argument('--print-shape', type=int, default=0)       # Model 1.
                                    # Print the data shape.

    parser.add_argument('--acc-test', action='store_true', help='Option for accuracy guard test in CH regression.')


def add_run_args(parser: argparse.ArgumentParser):
    """Add run arguments."""
    parser.add_argument('--data-folder',
                        type=str,
                        default='mnist_data',
                        help="The folder to download the MNIST dataset to.")
