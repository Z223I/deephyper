from deephyper.problem import HpProblem

Problem = HpProblem()


"""
(miniconda-3/latest/dl-hps) wilsonb@thetalogin6 model1:$ deephyper hps ambs --evaluator ray --problem model1.m1_hps.problem.Problem --run model1.m1_hps.model_run.run --n-jobs 1

Keras Version: 2.5.0
TIMER module loading: 0.0133 seconds
 ************************************************************************
   Maximizing the return value of function: model1.m1_hps.model_run.run
 ************************************************************************
2021-06-26 16:17:20,752	INFO services.py:1274 -- View the Ray dashboard at http://127.0.0.1:8265
Uncaught exception <class 'skopt.optimizer.optimizer.ExhaustedSearchSpace'>: The search space is exhausted and the search cannot sample new points!Traceback (most recent call last):
  File "/home/wilsonb/.conda/envs/dl-hps/bin/deephyper", line 33, in <module>
    sys.exit(load_entry_point('deephyper', 'console_scripts', 'deephyper')())
  File "/lus/theta-fs0/projects/datascience/wilsonb/theta/deephyper/deephyper/core/cli/cli.py", line 51, in main
    func(**kwargs)
  File "/lus/theta-fs0/projects/datascience/wilsonb/theta/deephyper/deephyper/core/cli/hps.py", line 43, in main
    search_obj.main()
  File "/lus/theta-fs0/projects/datascience/wilsonb/theta/deephyper/deephyper/search/hps/ambs.py", line 133, in main
    self.evaluator.add_eval_batch(self.get_random_batch(size=self.n_initial_points))
  File "/lus/theta-fs0/projects/datascience/wilsonb/theta/deephyper/deephyper/search/hps/ambs.py", line 220, in get_random_batch
    points = self.opt.ask(n_points=n_points)
  File "/home/wilsonb/.conda/envs/dl-hps/lib/python3.7/site-packages/skopt/optimizer/optimizer.py", line 443, in ask
    opt._tell(x, y_lie)
  File "/home/wilsonb/.conda/envs/dl-hps/lib/python3.7/site-packages/skopt/optimizer/optimizer.py", line 660, in _tell
    raise ExhaustedSearchSpace()
skopt.optimizer.optimizer.ExhaustedSearchSpace: The search space is exhausted and the search cannot sample new points!
"""

Problem.add_hyperparameter([10, 20], 'units')
"""
#Problem.add_hyperparameter(['relu', 'sigmoid', 'tanh'], 'activation')
#Problem.add_hyperparameter(['relu'], 'activation')
#Problem.add_hyperparameter(['Adam'], 'optimizer')
Problem.add_hyperparameter(['binary_crossentropy'], 'loss')
#Problem.add_hyperparameter([512, 1024, 2048, 4096] , 'batch_size')
Problem.add_hyperparameter([512, 1024] , 'batch_size')
Problem.add_hyperparameter([200] , 'epochs')
#Problem.add_hyperparameter((0.05, 0.055) , 'dropout')
Problem.add_hyperparameter([12], 'patience')
#Problem.add_hyperparameter([21], 'embed_hidden_size')
#Problem.add_hyperparameter([0.80, 0.85, 0.90, 0.95], 'proportion')
"""

"""
Problem.add_starting_point(
    units=10,
    activation='identity',
    lr=0.01)
"""

if __name__ == '__main__':
    print(Problem)