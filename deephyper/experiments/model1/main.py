from tensorflow import keras
print()
print(f'Keras Version: {keras.__version__}')

from deephyper.benchmarks_hps import util
timer = util.Timer()
timer.start("module loading")

import sys
import os

from keras import backend as K
from keras.models import Model
from keras.layers import Dense, Input, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

from datetime import datetime
import time

import numpy as np

from pprint import pprint
import tarfile
import math

from deephyper.benchmarks_hps.cliparser import build_base_parser

timer.end()

def getClassCount():
    """Return the number of classes."""
    return 2

from keras.activations import softmax

def softMaxAxis1(x):
    """Return softmax for axis 1."""
    return softmax(x, axis=1)


"""
import numpy as np
import tensorflow as tf
import random as rn
import os
os.environ["PYTHONHASHSEED"] = "0"
np.random.seed(42)
rn.seed(12345)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
from keras import backend as K
tf.set_random_seed(1234)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

from deephyper.benchmarks_hps import util
timer = util.Timer()
timer.start("module loading")

import keras
import re
import tarfile
import numpy as np
import math
from pprint import pprint
from functools import reduce
from keras.utils.data_utils import get_file
from keras.layers.embeddings import Embedding
from keras import layers
from keras.layers import recurrent
from keras.models import Model
from keras.callbacks import TerminateOnNaN, EarlyStopping
from keras.preprocessing.sequence import pad_sequences
from deephyper.benchmarks_hps.cliparser import build_base_parser
timer.end()
"""


def createModel(input_shape, samples, batchSamples, classCount, runNumber):
    """Create the Deep Learning model's graph.

    Arguments:
    input_shape -- shape of the input, usually (max_len,)

    Returns:
    model -- a model instance in Keras
    """
    # Define the input of the graph.
    inputLayer = Input(shape=input_shape, dtype='float32')

    X = inputLayer
    X = BatchNormalization(trainable=True)(X)

    # Propagate X through Dense layers

    if 20 <= batchSamples and False:
        X = Dense(units = samples * 20, activation='relu')(X)
        X = Dropout(0.20)(X)

    if 16 <= batchSamples:
        X = Dense(units = samples * 16, activation='relu')(X)
        X = Dropout(0.20)(X)

    if 12 <= batchSamples:
        X = Dense(units = samples * 12, activation='relu')(X)  # 12
        X = Dropout(0.10)(X)
        #X = Dense(units = samples * 11, activation='relu')(X)  # 11
        #X = Dropout(0.10)(X)

    if 10 <= batchSamples:
        X = Dense(units = samples * 10, activation='relu')(X)  # 10
        X = Dropout(0.05)(X)
        #X = Dense(units = samples * 9, activation='relu')(X)  # 9
        #X = Dense(units = samples * 8, activation='relu')(X)  # 8
        X = Dense(units = samples * 7, activation='relu')(X)  # 7

    if 5 <= batchSamples:
        X = Dense(units = samples * 5, activation='relu')(X)  # 5
        X = Dropout(0.05)(X)

    X = Dense(units = samples * 2, activation='relu')(X)  # 2
    X = Dropout(0.05)(X)
    X = Dense(units = 90, activation='relu')(X)

    #X = Dropout(0.05)(X)

    X = Dense(units = 90, activation='relu')(X)

    #X = Dropout(0.05)(X)

    #X = Dense(units = 48, activation='relu')(X)
    #X = Dropout(0.05)(X)
    X = Dense(units = 48, activation='relu')(X)

    #X = Dropout(0.05)(X)

    X = Dense(units = 24, activation='relu')(X)

    #X = Dropout(0.05)(X)

    #X = Dense(units = 24, activation='relu')(X)
    #X = Dropout(0.05)(X)
    X = Dense(units = 12, activation='relu')(X)

    #X = Dropout(0.10)(X)

    X = Dense(units = 12, activation='relu')(X)

    #X = Dropout(0.10)(X)

    X = Dense(units = 12, activation='relu')(X)
    #X = Dropout(0.05)(X)
    X = Dense(units = 5, activation='relu')(X)
    #X = Dropout(0.05)(X)
    #X = Dense(units = 5, activation='relu')(X)
    #X = Dropout(0.05)(X)
    #X = Dense(units = 5, activation='relu')(X)
    #X = Dropout(0.05)(X)
    X = Dense(units = 5, activation='relu')(X)
    #X = Dropout(0.05)(X)
    #X = Dense(units = 5, activation='relu')(X)
    #X = Dropout(0.05)(X)
    X = Dense(units = 5, activation='relu')(X)
    X = Dense(units = classCount, activation='softmax')(X)

    # Create Model instance which converts sentence_indices into X.
    model = Model(inputs=inputLayer, outputs=X)

    return model

def recall_m(y_true, y_pred):
    # sourcery skip: inline-immediately-returned-variable
    """Calculate recall."""
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    # sourcery skip: inline-immediately-returned-variable
    """Calculate precision."""
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    # sourcery skip: inline-immediately-returned-variable
    """Calculate F1."""
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    f1 = 2*((precision*recall)/(precision+recall+K.epsilon()))
    return f1






def get_data(f, only_supporting=False, max_length=None):
    '''Given a file name, read the file, retrieve the stories,
    and then convert the sentences into a single story.
    If max_length is supplied,
    any stories longer than max_length tokens will be discarded.
    '''
    data = parse_stories(f.readlines(), only_supporting=only_supporting)
    flatten = lambda data: reduce(lambda x, y: x + y, data)
    data = [(flatten(story), q, answer) for story, q, answer in data if not max_length or len(flatten(story)) < max_length]
    return data


def run(param_dict=None, verbose=2):
    """Run a param_dict on the reutersmlp benchmark."""
    # Read in values from CLI if no param dict was specified and clean up the param dict.
    param_dict = util.handle_cli(param_dict, build_parser())

    # Display the parsed param dict.
    if verbose:
        print("PARAM_DICT_CLEAN=")
        pprint(param_dict)

    if param_dict['data_source']:
        data_source = param_dict['data_source']
    else:
        data_source = os.path.dirname(os.path.abspath(__file__))
        data_source = os.path.join(data_source, 'data')

    path = os.path.join(data_source, "data.tar.gz")

    timer.start("data loading")
    with tarfile.open(path) as tar:
        train = get_data(tar.extractfile('train.txt'))
        test  = get_data(tar.extractfile('test.txt'))

    # print('vocab = {}'.format(vocab))
    # print('x.shape = {}'.format(x.shape))
    # print('xq.shape = {}'.format(xq.shape))
    # print('y.shape = {}'.format(y.shape))
    # print('story_maxlen, query_maxlen = {}, {}'.format(story_maxlen, query_maxlen))

    timer.end()

    BATCH_SIZE = param_dict['batch_size']
    ACTIVATION = util.get_activation_instance(param_dict['activation', param_dict['alpha']])
    EPOCHS = param_dict['epochs']
    DROPOUT = param_dict['dropout']
    OPTIMIZER = util.get_optimizer_instance(param_dict)

    #constants
    EMBED_HIDDEN_SIZE = 50
    patience = math.ceil(EPOCHS/2)

    # patient early stopping
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience = 12)

    callbacks = [ es, ]

    timer.start('preprocessing')

    # model_path = param_dict['model_path']
    # model_mda_path = None
    # model = None
    # initial_epoch = 0

    # if model_path:
    #     savedModel = util.resume_from_disk(BNAME, param_dict, data_dir=model_path)
    #     model_mda_path = savedModel.model_mda_path
    #     model_path = savedModel.model_path
    #     model = savedModel.model
    #     initial_epoch = savedModel.initial_epoch

    samples = 65
    batchSamples = 32

    numInputs = samples * batchSamples
    classCount = getClassCount()

    model = createModel((numInputs,), samples, batchSamples, classCount)
    model.compile(optimizer='AdamW', loss='binary_crossentropy', metrics=['acc', f1_m, precision_m, recall_m])

    timer.end()

    timer.start('model training')

    model.fit(Xtrain, Ytrain,
        batch_size = BATCH_SIZE,
        epochs=EPOCHS,
        shuffle=True,
        validation_data=(Xdev, Ydev),
        callbacks=callbacks,
        )

    # evaluate the model
    # TODO:  Why is loss used here.
    #loss, accuracy, f1_m, precision_m, recall_m = modelAnalyzeThis.evaluate(Xtest, Ytest, verbose=0)
    acc = model.evaluate(Xtest, Ytest, verbose=0)

    timer.end()
    return -acc[1]

def build_parser():  # sourcery skip: inline-immediately-returned-variable
    """Build this benchmark"s cli parser on top of the keras_cli parser."""
    parser = build_base_parser()

    return parser


if __name__ == "__main__":
    run()
