# Quickstart

## Model 1

### ThetaGPU

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

From thetagpusn1,2

```bash
qsub -I -A datascience -t 120 -q full-node -n 1
```


### Basic NAS Execution

```bash
cd /lus/theta-fs0/projects/datascience/wilsonb/theta/deephyper/model1/model1/m1_hps/
./SingleNodeRayCluster.sh
source ./SetUpEnv.sh
cd /lus/theta-fs0/projects/datascience/wilsonb/theta/deephyper/nas_problems/nas_problems/model1/
deephyper nas random --evaluator ray --problem problem.Problem

conda activate dl-hps
cd /lus/theta-fs0/projects/datascience/wilsonb/theta/deephyper/nas_problems/nas_problems/model1/problem.py
./SingleNodeRayCluster.sh
source ./SetUpEnv.sh
cd /lus/theta-fs0/projects/datascience/wilsonb/theta/deephyper
pip3 install -e .

deephyper nas random --evaluator ray --problem nas_problems.nas_problems.model1.problem.Problem

deephyper-analytics parse deephyper.log
Xdeephyper-analytics single -p $MY_JSON_FILE
Xdeephyper notebook --type nas --output mynotebook.ipynb $MY_JSON_FILE

deephyper-analytics notebook --type nas --output dh-analytics-nas.ipynb data_2021-07-14_01.json

```

I am attempting to run DeepHyper with NAS.  Because I am using my own model I had to do
a developer install for DeepHyper 0.2.5.

```bash
.../deephyper $ pip install -e .
```

I ran the model and parsed the log:

```bash
deephyper nas random --evaluator ray --problem nas_problems.nas_problems.model1.problem.Problem
deephyper-analytics parse deephyper.log
```

Then attempted the next step per https://deephyper.readthedocs.io/en/latest/tutorials/nas.html and
received the following error:

```bash
(conda/2021-06-26/base) wilsonb@thetagpu02:/lus/theta-fs0/projects/datascience/wilsonb/theta/deephyper$ deephyper-analytics single -p data_2021-07-14_01.json
detected env BALSAM_SPHINX_DOC_BUILD_ONLY: will not connect to a real DB
Module: 'balsam' module was found but not connected to a databse.
usage: deephyper-analytics [-h] {notebook,parse,quickplot,topk,balsam} ...
deephyper-analytics: error: invalid choice: 'single' (choose from 'notebook', 'parse', 'quickplot', 'topk', 'balsam')
```

Is single an old option?
