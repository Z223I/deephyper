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

from tensorflow import keras
print()
print(f'Keras Version: {keras.__version__}')

from deephyper.search.util import Timer
timer = Timer()
timer.start("module loading")

"""
      Successfully uninstalled Keras-2.4.3
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
tensorflow 2.5.0 requires h5py~=3.1.0, but you have h5py 3.3.0 which is incompatible.
tensorflow 2.5.0 requires numpy~=1.19.2, but you have numpy 1.21.0 which is incompatible.



(dhvenv) wilsonb@sm-01:~/deephyper/model1/model1/m1_hps$ pip install --force-reinstall Keras
Collecting Keras
  Using cached Keras-2.4.3-py2.py3-none-any.whl (36 kB)
Collecting pyyaml
  Using cached PyYAML-5.4.1-cp37-cp37m-manylinux1_x86_64.whl (636 kB)
Collecting scipy>=0.14
  Using cached scipy-1.7.0-cp37-cp37m-manylinux_2_5_x86_64.manylinux1_x86_64.whl (28.5 MB)
Collecting h5py
  Using cached h5py-3.3.0-cp37-cp37m-manylinux_2_12_x86_64.manylinux2010_x86_64.whl (4.1 MB)
Collecting numpy>=1.9.1
  Using cached numpy-1.21.0-cp37-cp37m-manylinux_2_12_x86_64.manylinux2010_x86_64.whl (15.7 MB)
Collecting cached-property
  Using cached cached_property-1.5.2-py2.py3-none-any.whl (7.6 kB)
Installing collected packages: numpy, cached-property, scipy, pyyaml, h5py, Keras
  Attempting uninstall: cached-property
    Found existing installation: cached-property 1.5.2
    Uninstalling cached-property-1.5.2:
      Successfully uninstalled cached-property-1.5.2
  Attempting uninstall: scipy
    Found existing installation: scipy 1.7.0
    Uninstalling scipy-1.7.0:
      Successfully uninstalled scipy-1.7.0
  Attempting uninstall: pyyaml
    Found existing installation: PyYAML 5.4.1
    Uninstalling PyYAML-5.4.1:
      Successfully uninstalled PyYAML-5.4.1
  Attempting uninstall: Keras
    Found existing installation: Keras 2.4.3
    Uninstalling Keras-2.4.3:
      Successfully uninstalled Keras-2.4.3
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
tensorflow 2.5.0 requires h5py~=3.1.0, but you have h5py 3.3.0 which is incompatible.
tensorflow 2.5.0 requires numpy~=1.19.2, but you have numpy 1.21.0 which is incompatible.
Successfully installed Keras-2.4.3 cached-property-1.5.2 h5py-3.3.0 numpy-1.21.0 pyyaml-5.4.1 scipy-1.7.0




  Successfully uninstalled Keras-2.4.3
(dhvenv) wilsonb@sm-01:~/deephyper/model1/model1/m1_hps$ pip install tensorflow
Requirement already satisfied: tensorflow in /lambda_stor/homes/wilsonb/venvs/dhvenv/lib/python3.7/site-packages (2.5.0)
Requirement already satisfied: astunparse~=1.6.3 in /lambda_stor/homes/wilsonb/venvs/dhvenv/lib/python3.7/site-packages (from tensorflow) (1.6.3)
Requirement already satisfied: gast==0.4.0 in /lambda_stor/homes/wilsonb/venvs/dhvenv/lib/python3.7/site-packages (from tensorflow) (0.4.0)
Requirement already satisfied: tensorboard~=2.5 in /lambda_stor/homes/wilsonb/venvs/dhvenv/lib/python3.7/site-packages (from tensorflow) (2.5.0)
Requirement already satisfied: google-pasta~=0.2 in /lambda_stor/homes/wilsonb/venvs/dhvenv/lib/python3.7/site-packages (from tensorflow) (0.2.0)
Requirement already satisfied: tensorflow-estimator<2.6.0,>=2.5.0rc0 in /lambda_stor/homes/wilsonb/venvs/dhvenv/lib/python3.7/site-packages (from tensorflow) (2.5.0)
Requirement already satisfied: keras-nightly~=2.5.0.dev in /lambda_stor/homes/wilsonb/venvs/dhvenv/lib/python3.7/site-packages (from tensorflow) (2.5.0.dev2021032900)
Requirement already satisfied: keras-preprocessing~=1.1.2 in /lambda_stor/homes/wilsonb/venvs/dhvenv/lib/python3.7/site-packages (from tensorflow) (1.1.2)
Collecting h5py~=3.1.0
  Using cached h5py-3.1.0-cp37-cp37m-manylinux1_x86_64.whl (4.0 MB)
Requirement already satisfied: wheel~=0.35 in /lambda_stor/homes/wilsonb/venvs/dhvenv/lib/python3.7/site-packages (from tensorflow) (0.36.2)
Requirement already satisfied: typing-extensions~=3.7.4 in /lambda_stor/homes/wilsonb/venvs/dhvenv/lib/python3.7/site-packages (from tensorflow) (3.7.4.3)
Requirement already satisfied: wrapt~=1.12.1 in /lambda_stor/homes/wilsonb/venvs/dhvenv/lib/python3.7/site-packages (from tensorflow) (1.12.1)
Requirement already satisfied: opt-einsum~=3.3.0 in /lambda_stor/homes/wilsonb/venvs/dhvenv/lib/python3.7/site-packages (from tensorflow) (3.3.0)
Requirement already satisfied: protobuf>=3.9.2 in /lambda_stor/homes/wilsonb/venvs/dhvenv/lib/python3.7/site-packages (from tensorflow) (3.17.3)
Requirement already satisfied: six~=1.15.0 in /lambda_stor/homes/wilsonb/venvs/dhvenv/lib/python3.7/site-packages (from tensorflow) (1.15.0)
Requirement already satisfied: termcolor~=1.1.0 in /lambda_stor/homes/wilsonb/venvs/dhvenv/lib/python3.7/site-packages (from tensorflow) (1.1.0)
Requirement already satisfied: grpcio~=1.34.0 in /lambda_stor/homes/wilsonb/venvs/dhvenv/lib/python3.7/site-packages (from tensorflow) (1.34.1)
Requirement already satisfied: absl-py~=0.10 in /lambda_stor/homes/wilsonb/venvs/dhvenv/lib/python3.7/site-packages (from tensorflow) (0.13.0)
Collecting numpy~=1.19.2
  Using cached numpy-1.19.5-cp37-cp37m-manylinux2010_x86_64.whl (14.8 MB)
Requirement already satisfied: flatbuffers~=1.12.0 in /lambda_stor/homes/wilsonb/venvs/dhvenv/lib/python3.7/site-packages (from tensorflow) (1.12)
Requirement already satisfied: cached-property in /lambda_stor/homes/wilsonb/venvs/dhvenv/lib/python3.7/site-packages (from h5py~=3.1.0->tensorflow) (1.5.2)
Requirement already satisfied: requests<3,>=2.21.0 in /lambda_stor/homes/wilsonb/venvs/dhvenv/lib/python3.7/site-packages (from tensorboard~=2.5->tensorflow) (2.25.1)
Requirement already satisfied: tensorboard-data-server<0.7.0,>=0.6.0 in /lambda_stor/homes/wilsonb/venvs/dhvenv/lib/python3.7/site-packages (from tensorboard~=2.5->tensorflow) (0.6.1)
Requirement already satisfied: tensorboard-plugin-wit>=1.6.0 in /lambda_stor/homes/wilsonb/venvs/dhvenv/lib/python3.7/site-packages (from tensorboard~=2.5->tensorflow) (1.8.0)
Requirement already satisfied: werkzeug>=0.11.15 in /lambda_stor/homes/wilsonb/venvs/dhvenv/lib/python3.7/site-packages (from tensorboard~=2.5->tensorflow) (2.0.1)
Requirement already satisfied: setuptools>=41.0.0 in /lambda_stor/homes/wilsonb/venvs/dhvenv/lib/python3.7/site-packages (from tensorboard~=2.5->tensorflow) (41.2.0)
Requirement already satisfied: google-auth<2,>=1.6.3 in /lambda_stor/homes/wilsonb/venvs/dhvenv/lib/python3.7/site-packages (from tensorboard~=2.5->tensorflow) (1.32.0)
Requirement already satisfied: markdown>=2.6.8 in /lambda_stor/homes/wilsonb/venvs/dhvenv/lib/python3.7/site-packages (from tensorboard~=2.5->tensorflow) (3.3.4)
Requirement already satisfied: google-auth-oauthlib<0.5,>=0.4.1 in /lambda_stor/homes/wilsonb/venvs/dhvenv/lib/python3.7/site-packages (from tensorboard~=2.5->tensorflow) (0.4.4)
Requirement already satisfied: cachetools<5.0,>=2.0.0 in /lambda_stor/homes/wilsonb/venvs/dhvenv/lib/python3.7/site-packages (from google-auth<2,>=1.6.3->tensorboard~=2.5->tensorflow) (4.2.2)
Requirement already satisfied: pyasn1-modules>=0.2.1 in /lambda_stor/homes/wilsonb/venvs/dhvenv/lib/python3.7/site-packages (from google-auth<2,>=1.6.3->tensorboard~=2.5->tensorflow) (0.2.8)
Requirement already satisfied: rsa<5,>=3.1.4 in /lambda_stor/homes/wilsonb/venvs/dhvenv/lib/python3.7/site-packages (from google-auth<2,>=1.6.3->tensorboard~=2.5->tensorflow) (4.7.2)
Requirement already satisfied: requests-oauthlib>=0.7.0 in /lambda_stor/homes/wilsonb/venvs/dhvenv/lib/python3.7/site-packages (from google-auth-oauthlib<0.5,>=0.4.1->tensorboard~=2.5->tensorflow) (1.3.0)
Requirement already satisfied: importlib-metadata in /lambda_stor/homes/wilsonb/venvs/dhvenv/lib/python3.7/site-packages (from markdown>=2.6.8->tensorboard~=2.5->tensorflow) (4.5.0)
Requirement already satisfied: pyasn1<0.5.0,>=0.4.6 in /lambda_stor/homes/wilsonb/venvs/dhvenv/lib/python3.7/site-packages (from pyasn1-modules>=0.2.1->google-auth<2,>=1.6.3->tensorboard~=2.5->tensorflow) (0.4.8)
Requirement already satisfied: urllib3<1.27,>=1.21.1 in /lambda_stor/homes/wilsonb/venvs/dhvenv/lib/python3.7/site-packages (from requests<3,>=2.21.0->tensorboard~=2.5->tensorflow) (1.26.6)
Requirement already satisfied: chardet<5,>=3.0.2 in /lambda_stor/homes/wilsonb/venvs/dhvenv/lib/python3.7/site-packages (from requests<3,>=2.21.0->tensorboard~=2.5->tensorflow) (4.0.0)
Requirement already satisfied: certifi>=2017.4.17 in /lambda_stor/homes/wilsonb/venvs/dhvenv/lib/python3.7/site-packages (from requests<3,>=2.21.0->tensorboard~=2.5->tensorflow) (2021.5.30)
Requirement already satisfied: idna<3,>=2.5 in /lambda_stor/homes/wilsonb/venvs/dhvenv/lib/python3.7/site-packages (from requests<3,>=2.21.0->tensorboard~=2.5->tensorflow) (2.10)
Requirement already satisfied: oauthlib>=3.0.0 in /lambda_stor/homes/wilsonb/venvs/dhvenv/lib/python3.7/site-packages (from requests-oauthlib>=0.7.0->google-auth-oauthlib<0.5,>=0.4.1->tensorboard~=2.5->tensorflow) (3.1.1)
Requirement already satisfied: zipp>=0.5 in /lambda_stor/homes/wilsonb/venvs/dhvenv/lib/python3.7/site-packages (from importlib-metadata->markdown>=2.6.8->tensorboard~=2.5->tensorflow) (3.4.1)
Installing collected packages: numpy, h5py
  Attempting uninstall: numpy
    Found existing installation: numpy 1.21.0
    Uninstalling numpy-1.21.0:
      Successfully uninstalled numpy-1.21.0
  Attempting uninstall: h5py
    Found existing installation: h5py 3.3.0
    Uninstalling h5py-3.3.0:
      Successfully uninstalled h5py-3.3.0
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
deephyper 0.2.5 requires keras, which is not installed.




(dhvenv) wilsonb@sm-01:~/deephyper/model1/model1/m1_hps$ pip install --no-cache-dir keras
Collecting keras
  Downloading Keras-2.4.3-py2.py3-none-any.whl (36 kB)
Requirement already satisfied: scipy>=0.14 in /lambda_stor/homes/wilsonb/venvs/dhvenv/lib/python3.7/site-packages (from keras) (1.7.0)
Requirement already satisfied: pyyaml in /lambda_stor/homes/wilsonb/venvs/dhvenv/lib/python3.7/site-packages (from keras) (5.4.1)
Requirement already satisfied: h5py in /lambda_stor/homes/wilsonb/venvs/dhvenv/lib/python3.7/site-packages (from keras) (3.1.0)
Requirement already satisfied: numpy>=1.9.1 in /lambda_stor/homes/wilsonb/venvs/dhvenv/lib/python3.7/site-packages (from keras) (1.19.5)
Requirement already satisfied: cached-property in /lambda_stor/homes/wilsonb/venvs/dhvenv/lib/python3.7/site-packages (from h5py->keras) (1.5.2)
Installing collected packages: keras
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
deephyper 0.2.5 requires tensorflow>=2.0.0, which is not installed.

compile --debug


"""
from keras import backend as K
from keras.models import Model
from keras.layers import Dense, Input, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping

#from pprint import pprint

timer.end("module loading")

def getClassCount():
    """Return the number of classes."""
    return 2

from keras.activations import softmax

def softMaxAxis1(x):
    """Return softmax for axis 1."""
    return softmax(x, axis=1)

from model1.m1_hps.load_data import load_data



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

    if batchSamples >= 16:
        X = Dense(units = samples * 16, activation='relu')(X)
        X = Dropout(0.20)(X)

    if batchSamples >= 12:
        X = Dense(units = samples * 12, activation='relu')(X)  # 12
        X = Dropout(0.10)(X)
        #X = Dense(units = samples * 11, activation='relu')(X)  # 11
        #X = Dropout(0.10)(X)

    if batchSamples >= 10:
        X = Dense(units = samples * 10, activation='relu')(X)  # 10
        X = Dropout(0.05)(X)
        #X = Dense(units = samples * 9, activation='relu')(X)  # 9
        #X = Dense(units = samples * 8, activation='relu')(X)  # 8
        X = Dense(units = samples * 7, activation='relu')(X)  # 7

    if batchSamples >= 5:
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


    timer.start('loading data')
    (x_train, y_train), (x_valid, y_valid) = load_data(config)
    timer.end('loading data')

    #
    # Retrieve config information.
    #
    BATCH_SIZE = config['batch_size']
    ACTIVATION = config['activation']
    EPOCHS = config['epochs']
    DROPOUT = config['dropout']
    OPTIMIZER = config['optimizer']

    #constants
    #EMBED_HIDDEN_SIZE = config['embed_hidden_size']
    PATIENCE = config['patience']

    # patient early stopping
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=PATIENCE)

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
    batchSamples = 26

    numInputs = samples * batchSamples
    classCount = getClassCount()

    model = createModel((numInputs,), samples, batchSamples, classCount)
    model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['acc', f1_m, precision_m, recall_m])

    timer.end('preprocessing')

    timer.start('model training')

    history = model.fit(x_train, y_train,
        batch_size = BATCH_SIZE,
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

    # evaluate the model
    # TODO:  Why is loss used here.
    #loss, accuracy, f1_m, precision_m, recall_m = modelAnalyzeThis.evaluate(Xtest, Ytest, verbose=0)
    #acc = model.evaluate(Xtest, yTest, verbose=0)

    timer.end('model training')

    HISTORY = history.history

    #return acc[-1]
    #return -acc[1]
    return history.history['acc'][-1]


if __name__ == '__main__':
    config = {
        #'units': 10,
        'activation': 'relu',  # can be gelu
        'optimizer':  'Adam',  # can be AdamW but it has to be installed.
        'loss':       'binary_crossentropy',
        'batch_size': 4096,
        'epochs':     200,
        'dropout':    0.05,
        'patience':   12,
        #'embed_hidden_size': 21,  # May not get used.
        #'proportion': .80           # A value between [0., 1.] indicating how to split data between
                                    # training set and validation set. `prop` corresponds to the
                                    # ratio of data in training set. `1.-prop` corresponds to the
                                    # amount of data in validation set.
    }
    accuracy = run(config)
    print('accuracy: ', accuracy)
    import matplotlib.pyplot as plt
    plt.plot(HISTORY['acc'])
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.show()