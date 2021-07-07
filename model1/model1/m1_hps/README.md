# Quickstart

## Model 1

### ThetaGPU

```bash
conda env remove --name dhgpu
conda remove --name dhgpu --all
rm -rf dhgpu
```

```bash
ssh wilsonb@theta.alcf.anl.gov
ssh wilsonb@thetagpusn1 or 2

xqsub -I -A $PROJECT_NAME -n 1 -t 60  Takes you to thetagpudd where dd are two numbers.

cd /lus/theta-fs0/projects/datascience/wilsonb/theta
xmodule load conda/2021-06-28
conda list --explicit > spec-file.txt
conda create --name dhgpu --file spec-file.txt

xconda create -p dhgpu --clone base
xchmod +r dhgpu/lib/python3.8/site-packages/easy-install.pth
conda activate dhgpu

xconda install 'tensorflow==2.5.0'
conda install tensorflow
xpip install --ignore-installed --upgrade tensorflow==2.5.0
conda install 'keras==2.4.3'

pip install pip --upgrade

I think this clashes with the deephyper in conda/2021-06-28
git clone https://github.com/deephyper/deephyper.git
cd deephyper/
git checkout develop


pip install -e .
```

```bash
```

Sam Foreman  7 hours ago
alternatively, it looks like you can use deephyper ray-submit directly from thetagpusn1 to automatically generate and submit a submission script
(documented at the very bottom of https://deephyper.readthedocs.io/en/develop/user_guides/thetagpu.html)

```bash
```

```bash
ssh thetagpusn1
export PROJECT_NAME=datascience
qsub -I -A $PROJECT_NAME -n 1 -t 30 -q full-node or single-gpu
cd /lus/theta-fs0/projects/datascience/wilsonb/theta/deephyper
deephyper nas random --evaluator ray --ray-address auto --problem deephyper.benchmark.nas.mnist1D.problem.Problem --max-evals 10 --num-cpus-per-task 1 --num-gpus-per-task 1
```

```bash
ssh wilsonb@theta.alcf.anl.gov
(miniconda-3/latest/base) wilsonb@thetalogin6:~> ssh thetagpusn1

Last login: Wed Jun 30 23:27:50 2021 from thetalogin4.tmi.alcf.anl.gov
wilsonb@thetagpusn1:~$ cd /lus/theta-fs0/projects/datascience/wilsonb/theta/
```

#### Start a node

```bash
Start skip
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
ssh -tt 127.0.1.1
source /lus/theta-fs0/projects/datascience/wilsonb/theta/deephyper/model1/model1/m1_hps/SetUpEnv.sh; ray start --head --node-ip-address=127.0.1.1 --port=6379     --num-cpus 8 --num-gpus 8 --block

[2]+  Stopped                 ssh -tt $head_node_ip "source $ACTIVATE_PYTHON_ENV;     ray start --head --node-ip-address=$head_node_ip --port=$port     --num-cpus $CPUS_PER_NODE --num-gpus $GPUS_PER_NODE --block"

End skip

From thetagpusn1,2
deephyper ray-submit hps ambs -n 1 -t 15 -A $PROJECT_NAME -q full-node --evaluator ray --problem model1.model1.m1_hps.problem.Problem --run model1.model1.m1_hps.model_run.run --n-jobs 2

```

### Another Try

Where to do the source?


```bash
source /lus/theta-fs0/software/thetagpu/conda/2021-06-28/mconda3/setup.sh
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

```bash
```

```bash
python model_run_pytorch.py
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
