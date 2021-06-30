# Quickstart

## Model 1

### ThetaGPU

```bash
ssh wilsonb@theta.alcf.anl.gov
ssh wilsonb@thetagpusn1 or 2
cd /lus/theta-fs0/projects/datascience/wilsonb/theta
module load conda/2021-06-28
conda create -p dhgpu2 --clone base
chmod +r dhgpu2/lib/python3.8/site-packages/easy-install.pth
conda activate /lus/theta-fs0/projects/datascience/wilsonb/theta/deephyper/dhgpu2
pip install pip --upgrade

I think this clashes with the deephyper in conda/2021-06-28
git clone https://github.com/deephyper/deephyper.git
cd deephyper/
git checkout develop
pip install -e .
```

#### Start a node

```bash
deephyper ray-submit nas agebo -w mnist_1gpu_2nodes_60 -n 2 -t 60 -A $PROJECT_NAME -q full-node --problem deephyper.benchmark.nas.mnist1D.problem.Problem --run deephyper.nas.run.alpha.run --max-evals 10000 --num-cpus-per-task 1 --num-gpus-per-task 1 -as $PATH_TO_SETUP --n-jobs 16

python -m deephyper.search.hps.ambs --evaluator ray --problem model1.model1.m1_hps.problem.Problem --run model1.model1.m1_hps.model_run.run --n-jobs 1

deephyper ray-submit hps ambs --evaluator ray --problem model1.m1_hps.problem.Problem --run model1.m1_hps.model_run.run --n-jobs 1
```

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
#deephyper hps ambs --evaluator ray --problem model1.m1_hps.problem.Problem --run model1.m1_hps.model_run.run --n-jobs 1
python -m deephyper.search.hps.ambs --evaluator ray --problem model1.model1.m1_hps.problem.Problem --run model1.model1.m1_hps.model_run.run --n-jobs 1
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
