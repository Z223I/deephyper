from deephyper.problem import NaProblem
from nas_problems.nas_problems.model1.load_data import load_data
from nas_problems.nas_problems.model1.search_space import create_search_space
from deephyper.nas.preprocessing import minmaxstdscaler

Problem = NaProblem(seed=2019)

Problem.load_data( load_data )

Problem.preprocessing(minmaxstdscaler)

Problem.search_space(create_search_space, num_layers=3)

Problem.hyperparameters(
    batch_size=32,
    learning_rate=0.01,
    optimizer='adam',
    num_epochs=20,
    callbacks=dict(
        EarlyStopping=dict(
            monitor='val_acc', # 'val_r2' or 'val_acc' ?
            mode='max',
            verbose=0,
            patience=5
        )
    )
)

Problem.loss('binary_crossentropy') # 'mse', 'binary_crossentropy' or 'categorical_crossentropy' ?

Problem.metrics(['acc']) # 'r2' or 'acc' ?

Problem.objective('val_acc__last') # 'val_r2__last' or 'val_acc__last' ?


# Get model.
if __name__ == '__main__':
    arch_seq = [0.8323024001314224, 0.7889290619064494, 0.9385678954207153, 0.16059637997392429,
    0.6488539744120456, 0.8325765404421139, 0.9888735139157468, 0.9322143923769549,
    0.7933123406870071]
    model = Problem.get_keras_model(arch_seq)

    print('Saving model...')
    model.save('model')

    json_config = model.to_json()
    #new_model = keras.models.model_from_json(json_config)


    import json

    jsonString = json.dumps(json_config)
    with open("model_to_json.json", "w") as jsonFile:
        jsonFile.write(jsonString)

    config = model.get_config()
    #new_model = keras.Sequential.from_config(config)

    jsonString = json.dumps(config)
    with open("model_get_config.json", "w") as jsonFile:
        jsonFile.write(jsonString)
