# Quickstart

## Model 1

### Basic Execution

```bash
conda activate dl-hps
cd /lus/theta-fs0/projects/datascience/wilsonb/theta/deephyper/model1/model1/m1_hps/
python model_run.py
```

## Hyperparameter Search (HPS)

An example command line for HPS:

```bash
#deephyper hps ambs --evaluator ray --problem model1.m1_hps.problem.Problem --run model1.m1_hps.model_run.run --n-jobs 1
deephyper hps ambs --evaluator ray --problem model1.m1_hps.problem.Problem --run model1.m1_hps.model_run.run --n-jobs 1
```

### Note

```bash
The above command will require a long time to execute completely. If you want to generate a smaller dataset, append `–max-evals 100’ to the end of the command to expedite the process.
```
