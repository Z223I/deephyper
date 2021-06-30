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

export PROJECT_NAME='datascience'
export PATH_TO_SETUP='/lus/theta-fs0/projects/datascience/wilsonb/theta/deephyper/model1/model1/m1_hps/'
source model1/model1/m1_hps/SingleNodeRayCluster.sh

(conda/2021-06-28//lus/theta-fs0/projects/datascience/wilsonb/theta/dhgpu) wilsonb@thetagpusn1:/lus/theta-fs0/projects/datascience/wilsonb/theta/deephyper$ source model1/model1/m1_hps/SingleNodeRayCluster.sh
Script to activate Python env: /lus/theta-fs0/projects/datascience/wilsonb/theta/deephyper/model1/model1/m1_hps/SetUpEnv.sh
thetagpusn1
>127.0.1.1<
IP Head: 127.0.1.1:6379
Starting HEAD at thetagpusn1

[1]+  Stopped                 ssh -tt $head_node_ip "source $ACTIVATE_PYTHON_ENV;     ray start --head --node-ip-address=$head_node_ip --port=$port     --num-cpus $CPUS_PER_NODE --num-gpus $GPUS_PER_NODE --block"



deephyper ray-submit hps ambs -n 1 -t 15 -A $PROJECT_NAME -q full-node --evaluator ray --problem model1.m1_hps.problem.Problem --run model1.m1_hps.model_run.run --n-jobs 2

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
