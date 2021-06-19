from tensorflow import keras
print()
print(f'Keras Version: {keras.__version__}')

import sys
import os

from keras.models import Model
from keras.layers import Dense, Input, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical

from datetime import datetime
import time

import numpy as np

def getClassCount():
    return 2

from keras.activations import softmax

def softMaxAxis1(x):
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
    """Function creating the DeepPlot model's graph.

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

    if runNumber >= 0:
       X = Dropout(0.05)(X)

    X = Dense(units = 90, activation='relu')(X)

    if runNumber >= 0:
       X = Dropout(0.05)(X)

    #X = Dense(units = 48, activation='relu')(X)
    #X = Dropout(0.05)(X)
    X = Dense(units = 48, activation='relu')(X)

    if runNumber >= 1:
       X = Dropout(0.05)(X)

    X = Dense(units = 24, activation='relu')(X)

    if runNumber >= 1:
       X = Dropout(0.05)(X)

    #X = Dense(units = 24, activation='relu')(X)
    #X = Dropout(0.05)(X)
    X = Dense(units = 12, activation='relu')(X)

    if runNumber >= 2:
       X = Dropout(0.10)(X)

    X = Dense(units = 12, activation='relu')(X)

    if runNumber >= 2:
       X = Dropout(0.10)(X)

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

    # Create Model
    model = Model(inputs=inputLayer, outputs=X)

    return model






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

    try:
        path = get_file('babi-tasks-v1-2.tar.gz', origin='https://s3.amazonaws.com/text-datasets/babi_tasks_1-20_v1-2.tar.gz')
    except:
        print('Error downloading dataset, please download it manually:\n'
            '$ wget http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz\n'
            '$ mv tasks_1-20_v1-2.tar.gz ~/.keras/datasets/babi-tasks-v1-2.tar.gz')
        raise

    challenge = 'tasks_1-20_v1-2/en-10k/qa2_two-supporting-facts_{}.txt'

    with tarfile.open(path) as tar:
        train = get_stories(tar.extractfile(challenge.format('train')))
        test = get_stories(tar.extractfile(challenge.format('test')))

    vocab = set()
    for story, q, answer in train + test:
        vocab |= set(story + q + [answer])
    vocab = sorted(vocab)

    vocab_size = len(vocab) + 1
    word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
    story_maxlen = max(map(len, (x for x, _, _ in train + test)))
    query_maxlen = max(map(len, (x for _, x, _ in train + test)))

    x, xq, y = vectorize_stories(train, word_idx, story_maxlen, query_maxlen)
    tx, txq, ty = vectorize_stories(test, word_idx, story_maxlen, query_maxlen)

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
    callbacks = [
    EarlyStopping(monitor="val_acc", min_delta=0.0001, patience=patience, verbose=verbose, mode="auto"),
    TerminateOnNaN()]

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

    # if model is None:
    sentence = layers.Input(shape=(story_maxlen,), dtype='int32')
    encoded_sentence = layers.Embedding(vocab_size,  EMBED_HIDDEN_SIZE)(sentence)
    encoded_sentence = layers.Dropout(DROPOUT)(encoded_sentence)
    question = layers.Input(shape=(query_maxlen,), dtype='int32')
    encoded_question = layers.Embedding(vocab_size,  EMBED_HIDDEN_SIZE)(question)
    encoded_question = layers.Dropout(DROPOUT)(encoded_question)
    encoded_question = RNN( EMBED_HIDDEN_SIZE)(encoded_question)
    encoded_question = layers.RepeatVector(story_maxlen)(encoded_question)

    merged = layers.add([encoded_sentence, encoded_question])
    merged = RNN( EMBED_HIDDEN_SIZE)(merged)
    merged = layers.Dropout(DROPOUT)(merged)
    preds = layers.Dense(vocab_size, activation=ACTIVATION)(merged)

    model = Model([sentence, question], preds)
    model.compile(optimizer=OPTIMIZER,
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    timer.end()

    timer.start('model training')

    model.fit([x, xq], y,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            callbacks=callbacks,
            validation_split=0.05)
    acc = model.evaluate([tx, txq], ty,
                            batch_size=BATCH_SIZE)

    timer.end()
    return -acc[1]

def build_parser():  # sourcery skip: inline-immediately-returned-variable
    """Build this benchmark"s cli parser on top of the keras_cli parser."""
    parser = build_base_parser()

    return parser


if __name__ == "__main__":
    run()
