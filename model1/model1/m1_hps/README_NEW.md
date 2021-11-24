# Recreating HPS

## Test model

This works.  Only tests model.  There is no HPS.

```bash
dlvenv
cd model1/model1/m1_hps/
python model_run_keras.py 
```

## Run Hyperparameter Search (HPS)

You must be in the correct directory.

```bash
dlvenv
cd ~/DL/deephyper
git checkout feature/HPS-006B-deephyper
cd ./model1/model1/m1_hps/
python -m deephyper.search.hps.ambs --evaluator ray --problem model1.model1.m1_hps.problem.Problem --run model1.model1.m1_hps.model_run.run --n-jobs 1 --max-evals 5
```

## Hyperparameter Search (HPS)

An example command line for HPS:

```bash
python -m deephyper.search.hps.ambs --evaluator ray --problem model1.model1.m1_hps.problem.Problem --run model1.model1.m1_hps.model_run.run --n-jobs 1
```


