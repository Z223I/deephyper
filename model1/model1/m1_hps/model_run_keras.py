"""Perform basic testing with the HyperParameter package."""

"""
Usage
deephyper hps ambs --evaluator ray --problem deephyper.benchmark.hps.polynome2.Problem --run deephyper.m1_hps.run --n-jobs 1

python -m deephyper.search.hps.ambs --evaluator threadPool --problem deephyper.model1.model1.m1_hps.Problem --run deephyper.model1.model1.m1_hps.run --max-evals 100 --kappa 0.001

python -m model_run.py
"""

"""
Example from deephyper.benchmark.hps.polynome2
python -m deephyper.search.hps.ambs2 --evaluator threadPool --problem deephyper.benchmark.hps.polynome2.Problem --run deephyper.benchmark.hps.polynome2.run --max-evals 100 --kappa 0.001

I have seen:
    --evaluator threadPool
    and
    --evaluator ray
"""

from tensorflow import keras as tf_keras
print(f'TensorFlow.Keras Version: {tf_keras.__version__}')

#####from deephyper.search.util import Timer
#####timer = Timer()
#####timer.start("module loading")

from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow

#from pprint import pprint

#####timer.end("module loading")

def getClassCount():
    """Return the number of classes."""
    return 2

from tensorflow.keras.activations import softmax

def softMaxAxis1(x):
    """Return softmax for axis 1."""
    return softmax(x, axis=1)

from load_data import load_data



def createModel(input_shape, samples, batchSamples, classCount):
    # sourcery skip: inline-immediately-returned-variable
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

    X = Dense(units = samples * 16, activation='relu')(X)
    X = Dropout(0.20)(X)

    X = Dense(units = samples * 12, activation='relu')(X)  # 12
    X = Dropout(0.10)(X)
    #X = Dense(units = samples * 11, activation='relu')(X)  # 11
    #X = Dropout(0.10)(X)

    X = Dense(units = samples * 10, activation='relu')(X)  # 10
    X = Dropout(0.05)(X)
    #X = Dense(units = samples * 9, activation='relu')(X)  # 9
    #X = Dense(units = samples * 8, activation='relu')(X)  # 8
    X = Dense(units = samples * 7, activation='relu')(X)  # 7

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

HISTORY = None


def run(config):
    """Run model."""
    global HISTORY


    #####timer.start('loading data')
    (x_train, y_train), (x_valid, y_valid) = load_data(config)
    #####timer.end('loading data')

    #
    # Retrieve config information.
    #
    BATCH_SIZE = config['batch_size']
    LOSS       = config['loss']
    ACTIVATION = config['activation']
    EPOCHS = config['epochs']
    #DROPOUT = config['dropout']
    OPTIMIZER = config['optimizer']

    #constants
    EMBED_HIDDEN_SIZE = config['embed_hidden_size']
    PATIENCE = config['patience']

    # patient early stopping
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=PATIENCE)

    callbacks = [ es, ]

    #####timer.start('preprocessing')

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
    batchSamples = 26

    numInputs = samples * batchSamples
    classCount = getClassCount()

    doCreateModel = True
    if doCreateModel:
        model = createModel((numInputs,), samples, batchSamples, classCount, config)
    else:
        # Use model that NAS built.
        model = tensorflow.keras.models.load_model('model')
        #model = keras.models.load_model("../../../nas_problems/nas_problems/model1/model.h5")

    #model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=[['acc'], [f1_m], [precision_m], [recall_m]])
    model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=['acc'])

    #####timer.end('preprocessing')

    #####timer.start('model training')

    history = model.fit(x_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        shuffle=True,
        validation_data=(x_valid, y_valid),
        callbacks=callbacks,
        )

    """
    # Try this sometime.
    model.fit(x_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        shuffle=True,
        callbacks=callbacks,
        validation_split=0.05)
    """

    MISC_STUFF = False
    if MISC_STUFF:
        # "You have to run the model for at least 1 epoch for the metric names to be available"
        metrics = model.metrics_names
        print(f"metrics: {metrics}")

        #####timer.end('model training')

        model.summary()
        model.save('model_keras')

        HISTORY = history.history

    return history.history['acc'][-1]


if __name__ == '__main__':
    config = {
        'units': 10,
        'activation': 'relu',  # can be gelu
        'optimizer':  'Adam',  # can be AdamW but it has to be installed.
        'loss':       'binary_crossentropy',
        'batch_size': 32,
        'epochs':     30,
        'dropout':    0.05,
        'patience':   12,
        'embed_hidden_size': 21,    # May not get used.
        'proportion': .80           # A value between [0., 1.] indicating how to split data between
                                    # training set and validation set. `prop` corresponds to the
                                    # ratio of data in training set. `1.-prop` corresponds to the
                                    # amount of data in validation set.
    }
    accuracy = run(config)
    print('accuracy: ', accuracy)

    """
    import matplotlib.pyplot as plt
    plt.plot(HISTORY['acc'])
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.show()
    """
