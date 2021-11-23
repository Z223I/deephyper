# Quickstart HPS Model 1.2

## ssh to Login Node

```bash
theta
```

Password: < Enter your eight digit code from the MobilPass+ app >

## Git as Necessary

```bash
cd /lus/theta-fs0/projects/datascience/wilsonb/theta/deephyper
git pull
```

## ssh to thetagpusn1

```bash
ssh thetagpusn1
```

## Request Time for a ThetaGPU Node

```bash
export PROJECT_NAME=datascience
qsub -I -A $PROJECT_NAME -n 1 -t 30 -q full-node
```

## thetagpu

### Start Ray Cluster

```bash
cd /lus/theta-fs0/projects/datascience/wilsonb/theta/deephyper/model1/model1/m1_hps/
./SingleNodeRayCluster.sh
```

Wait until

```text
--block
  This command will now block until terminated by a signal.
  Running subprocesses are monitored and a message will be printed if any of them terminate unexpectedly.
```

### Start Conda Env

```bash
module load conda/2021-06-26
source ./SetUpEnv.sh
```

### Run Model

This is a check to ensure your model is running correctly.

```bash
python3 model_run_keras.py
```

### Run DeepHyper

```bash
python -m deephyper.search.hps.ambs --evaluator ray --problem problem.Problem --run model_run_keras.run --num-cpus-per-task 1 --num-gpus-per-task 1 --n-jobs 1
```

### Note

```bash
The above command will require a long time to execute completely. If you want to generate a smaller dataset, append '--max-evals 100’ to the end of the command to expedite the process.
```

## Analytics

Balsam is needed for Analytics.  Please see the documentation at https://balsam.readthedocs.io/en/latest/ for installation information.

```bash
deephyper-analytics notebook --type hps --output dh-analytics-hps.ipynb results.csv
```
