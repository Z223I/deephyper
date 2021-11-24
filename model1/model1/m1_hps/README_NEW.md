# Recreating HPS

## History

Did not work.

```bash
dlvenv
git checkout feature/006B-deephyper
python -m deephyper.search.hps.ambs --evaluator ray --problem model1.model1.m1_hps.problem.Problem --run model1.model1.m1_hps.model_run.run --n-jobs 1
```

## Test model

This works.  Only tests model.  There is no HPS.

```bash
dlvenv
cd model1/model1/m1_hps/
python model_run_keras.py 
```

