from deephyper.problem import HpProblem

Problem = HpProblem()

#Problem.add_hyperparameter(['relu', 'sigmoid', 'tanh'], 'activation')
Problem.add_hyperparameter(['relu', 'gelu', 'tanh'], 'activation')
Problem.add_hyperparameter(['Adam', 'AdamW'], 'optimizer')
Problem.add_hyperparameter(['binary_crossentropy'], 'loss')
Problem.add_hyperparameter([256, 512, 1024], 'batch_size')
Problem.add_hyperparameter([100], 'epochs')
Problem.add_hyperparameter([0.10, 0.15, 0.20], 'dropout1')
Problem.add_hyperparameter([0.05, 0.10, 0.15], 'dropout2')
Problem.add_hyperparameter([0.0, 0.05, 0.10], 'dropout3')
Problem.add_hyperparameter([0.0, 0.05], 'dropout4')
Problem.add_hyperparameter([20], 'patience')
Problem.add_hyperparameter([21], 'embed_hidden_size')
Problem.add_hyperparameter([0.90, 0.95], 'proportion')

"""
Problem.add_starting_point(
    units=10,
    activation='identity',
    lr=0.01)
"""

if __name__ == '__main__':
    print(Problem)
