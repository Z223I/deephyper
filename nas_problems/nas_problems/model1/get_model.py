from deephyper.problem import NaProblem

Problem = NaProblem(seed=2019)

# Just to print your problem, to test its definition and imports in the current python environment.
if __name__ == '__main__':
    arch_seq = [0.8323024001314224, 0.7889290619064494, 0.9385678954207153, 0.16059637997392429,
    0.6488539744120456, 0.8325765404421139, 0.9888735139157468, 0.9322143923769549,
    0.7933123406870071]
    model = Problem.get_keras_model(arch_seq)

    print('Saving model...')
    model.save('model')
