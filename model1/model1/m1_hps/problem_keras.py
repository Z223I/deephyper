from deephyper.problem import HpProblem

Problem = HpProblem()

Problem.add_hyperparameter([10], 'units')
#Problem.add_hyperparameter(['relu', 'gelu', 'sigmoid', 'tanh'], 'activation')
Problem.add_hyperparameter(['relu'], 'activation')
Problem.add_hyperparameter(['Adam', 'AdamW'], 'optimizer')
Problem.add_hyperparameter(['binary_crossentropy'], 'loss')
Problem.add_hyperparameter([32, 64, 128], 'batch_size')
Problem.add_hyperparameter([75, 150], 'epochs')
Problem.add_hyperparameter([0.05, 0.10], 'dropout1')
Problem.add_hyperparameter([0.05, 0.10], 'dropout2')
Problem.add_hyperparameter([0.05, 0.10], 'dropout3')
Problem.add_hyperparameter([0.05], 'dropout4')
Problem.add_hyperparameter([12], 'patience')
Problem.add_hyperparameter([21], 'embed_hidden_size')
Problem.add_hyperparameter([0.50], 'proportion')

"""
Problem.add_starting_point(
    units=10,
    activation='identity',
    lr=0.01)
"""

if __name__ == '__main__':
    print(Problem)