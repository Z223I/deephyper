# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'


# %% [markdown]
# Let's get started! Run the following cell to load the package you are going to use. 

# %%
import keras
print()
print(f'Keras Version: {keras.__version__}')

import sys
import os

#get_ipython().run_line_magic('matplotlib', 'inline')


# %%
from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation, BatchNormalization

from datetime import datetime
#from packaging import version

# %% [markdown]
# ## 2.3 Building the Emojifier-V2
# 
# Lets now build the Emojifier-V2 model. 
# * You feed the embedding layer's output to an LSTM network. 
# 
# <img src="images/emojifier-v2.png" style="width:700px;height:400px;"> <br>
# <caption><center> **Figure 3**: Emojifier-v2. A 2-layer LSTM sequence classifier. </center></caption>
# 
# 
# **Exercise:** Implement `Emojify_V2()`, which builds a Keras graph of the architecture shown in Figure 3. 
# * The model takes as input an array of sentences of shape (`m`, `max_len`, ) defined by `input_shape`. 
# * The model outputs a softmax probability vector of shape (`m`, `C = 5`). 
# 
# * You may need to use the following Keras layers:
#     * [Input()](https://keras.io/layers/core/#input)
#         * Set the `shape` and `dtype` parameters.
#         * The inputs are integers, so you can specify the data type as a string, 'int32'.
#     * [LSTM()](https://keras.io/layers/recurrent/#lstm)
#         * Set the `units` and `return_sequences` parameters.
#     * [Dropout()](https://keras.io/layers/core/#dropout)
#         * Set the `rate` parameter.
#     * [Dense()](https://keras.io/layers/core/#dense)
#         * Set the `units`, 
#         * Note that `Dense()` has an `activation` parameter.  For the purposes of passing the autograder, please do not set the activation within `Dense()`.  Use the separate `Activation` layer to do so.
#     * [Activation()](https://keras.io/activations/).
#         * You can pass in the activation of your choice as a lowercase string.
#     * [Model](https://keras.io/models/model/)
#         Set `inputs` and `outputs`.
# 
# 
# #### Additional Hints
# * Remember that these Keras layers return an object, and you will feed in the outputs of the previous layer as the input arguments to that object.  The returned object can be created and called in the same line.
# 
# ```Python
# # How to use Keras layers in two lines of code
# dense_object = Dense(units = ...)
# X = dense_object(inputs)
# 
# # How to use Keras layers in one line of code
# X = Dense(units = ...)(inputs)
# ```
# %%
#>>> # Create a `Sequential` model and add a Dense layer as the first layer.  
#>>> model = tf.keras.models.Sequential()
#>>> model.add(tf.keras.Input(shape=(16,)))
#>>> model.add(tf.keras.layers.Dense(32, activation='relu'))
#>>> # Now the model will take as input arrays of shape (None, 16)  
#>>> # and output arrays of shape (None, 32).  
#>>> # Note that after the first layer, you don't need to specify  
#>>> # the size of the input anymore:  
#>>> model.add(tf.keras.layers.Dense(32))
#>>> model.output_shape
#(None, 32)
# ```

# %%
import numpy as np

def getClassCount():
    return 2

# %%
from keras.activations import softmax

def softMaxAxis1(x):
    return softmax(x,axis=1)

def createAnalyzeThisModel(input_shape, samples, samplesPerDay, days, classCount):
    """
    Function creating the DeepPlot model's graph.
    
    Arguments:
    input_shape -- shape of the input, usually (max_len,)

    Returns:
    model -- a model instance in Keras
    """

    # Define the input of the graph.
    # It should be of shape input_shape and dtype 'int32' (as it contains indices, which are integers).
    inputLayer = Input(shape=input_shape, dtype='float32')


    #X = Dense(units = 180, activation='relu', kernel_regularizer=keras.regularizers.l2(1.00))(X)

    X = inputLayer
    X = BatchNormalization(trainable=True)(X)

    # Propagate X througeh Dense layers

    if 20 <= days and False:
        X = Dense(units = samples * samplesPerDay * 20, activation='relu')(X)
        X = Dropout(0.20)(X)

    if 16 <= days:
        X = Dense(units = samples * samplesPerDay * 16, activation='relu')(X)
        X = Dropout(0.20)(X)

    if 12 <= days:
        X = Dense(units = samples * samplesPerDay * 12, activation='relu')(X)  # 12
        X = Dropout(0.10)(X)
        #X = Dense(units = samples * samplesPerDay * 11, activation='relu')(X)  # 11
        #X = Dropout(0.10)(X)

    if 10 <= days:
        X = Dense(units = samples * samplesPerDay * 10, activation='relu')(X)  # 10
        X = Dropout(0.05)(X)
        #X = Dense(units = samples * samplesPerDay * 9, activation='relu')(X)  # 9
        #X = Dense(units = samples * samplesPerDay * 8, activation='relu')(X)  # 8
        X = Dense(units = samples * samplesPerDay * 7, activation='relu')(X)  # 7

    if 5 <= days:
        X = Dense(units = samples * samplesPerDay * 5, activation='relu')(X)  # 5
        #X = Dropout(0.05)(X)

    X = Dense(units = samples * samplesPerDay * 2, activation='relu')(X)  # 2
    #X = Dropout(0.05)(X)
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
    #X = Dropout(0.05)(X)
    X = Dense(units = 12, activation='relu')(X)
    #X = Dropout(0.05)(X)
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

# %% [markdown]
# Generate Model Summary

# %%
logdir = 'logs/scalars/' + datetime.now().strftime('%Y%m%d_%H%M%S')
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

# %%

from config import AtConfig
atConfig = AtConfig()

if atConfig.predictionOnly:
    print('ERROR: Turn off prediction and regenerate files.')
    sys.exit(-1)

samples = atConfig.numSamples
samplesPerDay = atConfig.samplesPerDay
days = atConfig.durationHistorical

numInputs = samples * samplesPerDay * days
classCount = getClassCount()

modelAnalyzeThis = createAnalyzeThisModel((numInputs,), samples, samplesPerDay, days, classCount)
modelAnalyzeThis.summary()






from keras.callbacks import EarlyStopping

# patient early stopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience = 12)

from keras import backend as K

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

# compile the model
#modelAnalyzeThis.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc',f1_m,precision_m, recall_m])
modelAnalyzeThis.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#[plots, plotLabels, numPlots] = executableMain()
baseDirectory = './'

#coresToUse = 6
#K.set_session(K.tf.Session(config=K.tensorflow.compat.v1.ConfigProto(intra_op_‌​parallelism_threads=coresToUse, inter_op_parallelism_threads=coresToUse)))

Xtrain = np.loadtxt(f"{baseDirectory}aws/XTrain.at", delimiter=",")
Ytrain = np.loadtxt(f"{baseDirectory}aws/YTrain.at", delimiter=",", dtype=np.int32)
Ytrain = keras.utils.to_categorical(Ytrain)

print(f'Xtrain shape: {Xtrain.shape}')
print(f'Ytrain shape: {Ytrain.shape}')

countClasses = Ytrain.shape[1]
print(f'Classes: {countClasses}')
print()
print('tensorboard --logdir logs/scalars')
print('http://localhost:6006/')

Xdev = np.loadtxt(f"{baseDirectory}aws/XDev.at", delimiter=",")
Ydev = np.loadtxt(f"{baseDirectory}aws/YDev.at", delimiter=",")
Ydev = keras.utils.to_categorical(Ydev)

print(f'Xdev shape: {Xdev.shape}')
print(f'Ydev shape: {Ydev.shape}')

numEpochs = 200

history = modelAnalyzeThis.fit(Xtrain, Ytrain, epochs = numEpochs, \
    batch_size = 4096, shuffle=True, 
    validation_data=(Xdev, Ydev),
    callbacks=[tensorboard_callback, es],
    )

# fit the model
#history = model.fit(Xtrain, ytrain, validation_split=0.3, epochs=10, verbose=0)

#[plots, plotLabels, numPlots] = executableMain()
if os.path.isdir('home/ubuntu'):
    baseDirectory = '/home/ubuntu/DL/Stocks/analyzethis/'
else:
    baseDirectory = './'


#Xtest = np.loadtxt(f"{baseDirectory}aws/XTest.at", delimiter=",")
#Ytest = np.loadtxt(f"{baseDirectory}aws/YTest.at", delimiter=",")
#Ytest = keras.utils.to_categorical(Ytest)

# evaluate the model
#loss, accuracy, f1_score, precision, recall = modelAnalyzeThis.evaluate(Xtest, Ytest, verbose=0)
#accuracy = modelAnalyzeThis.evaluate(Xtest, Ytest, verbose=0)

# Predict using Xtest.
#predictions = model.predict(Xtest)


# %%
print('Saving model...')
modelAnalyzeThis.save('model')

