"""Problem description."""

from deephyper.problem import NaProblem
from nas_problems.nas_problems.model1.load_data import load_data
from nas_problems.nas_problems.model1.search_space import create_search_space
from deephyper.nas.preprocessing import minmaxstdscaler

Problem = NaProblem(seed=2019)

Problem.load_data( load_data )

# 20211127 Do not scale.  AT does not scale.
#Problem.preprocessing(minmaxstdscaler)

Problem.search_space(create_search_space, num_layers=6)

# Metrics: [metrics.mae, metrics.accuracy, metrics.binary_accuracy, metrics.categorical_accuracy, metrics.sparse_categorical_accuracy, metrics.confusion_matrix]

Problem.hyperparameters(
    batch_size=256,
    learning_rate=0.01,
    optimizer='adamw',
    num_epochs=75,
    callbacks=dict(
        ModelCheckpoint=dict(
                        monitor="val_loss",
                        mode="min",
                        save_best_only=True,
                        verbose=0,
                        filepath="model_nas_001",
                        save_weights_only=False,
                    ),
        EarlyStopping=dict(
            monitor='val_loss', # 'val_loss', 'val_r2' or 'val_acc' ?
            mode='min',
            verbose=0,
            patience=12
        )
    )
)


Problem.loss('binary_crossentropy') # 'mse', 'binary_crossentropy' or 'categorical_crossentropy' ?

# Metrics: accuracy and validation loss.
#Problem.metrics(['acc', 'val_loss']) # 'r2', 'acc', 'val_loss'
Problem.metrics(['acc']) # 'r2', 'acc', 'val_loss'

#Problem.objective('val_loss__last') # 'val_r2__last', 'val_acc__last'
# Using val_loss__min instead of val_loss__last will cause validation to be calculated for every
# epoch.  This is required to enable early stopping.
Problem.objective('-val_loss') # 'val_r2__last', 'val_acc__last'


# Get model.
if __name__ == '__main__':
    arch_seq = [0.25115011042544255, 0.008666602895180286, 0.574007524631404, 0.8273919005340072,
    0.6361905672637143, 0.6318902366378124, 0.20937749329603517, 0.7575470589113212,
    0.4126362184199489]
    model = Problem.get_keras_model(arch_seq)

    model.summary()
    optimizer = 'adam'
    loss = 'binary_crossentropy'
    model.compile(optimizer, loss, metrics=['accuracy'])

    (XTrain, YTrain), (XValid, YValid) = load_data()
    accuracy = model.evaluate(XTrain, YTrain, verbose=1)
    print(f'Train accuracy: {accuracy}')
    accuracy = model.evaluate(XValid, YValid, verbose=1)
    print(f'Validation accuracy: {accuracy}')

    print('Saving model...')
    model.save('model_nas_001')

    ## This is needed for converting from Keras to PyTorch.
    model.save_weights('model.h5')

    #
    # Save model config info.
    #

    print('Saving model.json...')
    ## This is needed for converting from Keras to PyTorch.
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)






    import tensorflow.keras as keras
    # From https://www.tensorflow.org/guide/keras/save_and_serialize
    # Calling `save('my_model')` creates a SavedModel folder `my_model`.
    model.save("my_model")

    # It can be used to reconstruct the model identically.
    reconstructed_model = keras.models.load_model("my_model")


    """
    import onnx

    pytorch_model = '/path/to/pytorch/model'
    keras_output = '/path/to/converted/keras/model.hdf5'
    onnx.convert(pytorch_model, keras_output)
    reconstructed_model = load_model(keras_output)
    preds = reconstructed_model.predict(x)
    """
