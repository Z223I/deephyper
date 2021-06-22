from deephyper.problem import HpProblem

Problem = HpProblem()

Problem.add_hyperparameter('(10) , units')
#Problem.add_hyperparameter(['relu', 'sigmoid', 'tanh'], 'activation')
Problem.add_hyperparameter(['relu'], 'activation')
Problem.add_hyperparameter(['Adam'], 'optimizer')
Problem.add_hyperparameter(['binary_crossentropy'], 'loss')
Problem.add_hyperparameter((4096) , 'batch_size')
Problem.add_hyperparameter((200) , 'epochs')
Problem.add_hyperparameter((0.05) , 'dropout')
Problem.add_hyperparameter((12), 'patience')
Problem.add_hyperparameter((21), 'embed_hidden_size')
Problem.add_hyperparameter((0.80), 'proportion')

"""
Problem.add_starting_point(
    units=10,
    activation='identity',
    lr=0.01)
"""

if __name__ == '__main__':
    print(Problem)