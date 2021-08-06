# Quickstart NAS Model 1

```text
I know how to do it but I don’t have the code ready for you, it needs to be developed in python basically you can use the __code__  attribute to retrieve the code

and so instead of creating the tensors the same recursive function can be called to construct the code
```

```text
https://github.com/deephyper/deephyper/blob/develop/deephyper/nas/space/keras_search_space.py#L119
def create_model(self):

an[d] call the recursive function create_tensor_aux

Romain  2 minutes ago
https://github.com/deephyper/deephyper/blob/74dcc25245cc1c6eda4a547e3a24f1704945f9ae/deephyper/nas/space/nx_search_space.py#L180
deephyper/nas/space/nx_search_space.py:180
    def create_tensor_aux(self, g, n, train=None):
<https://github.com/deephyper/deephyper|deephyper/deephyper>deephyper/deephyper | Added by GitHub (Legacy)


and this recursive function calls n.create_tensor

what you want is the __code__ of n.create_tensor
```

## ssh to homes

```bash
$ homes
or
$ ssh <username>@homes.anl.gov
```

## ssh to sm-01

You must use sm-01 or change the IP address below.

```bash
sm1
```

## SM-01

```bash
snp
```

## Activate or Create Venv

```bash
source ~/venvs/dhvenv4/bin/activate
```

```bash
virtualenv --system-site-packages -p python3.7 ./venv
source ./venv/bin/activate
pip install --find-links=https://download.pytorch.org/whl/torch_stable.html -r /opt/sambaflow/apps/requirements.txt
```

## Git as Necessary

```bash
git clone https://github.com/deephyper/deephyper.git
or
cd ~/deephyper
git pull
```

## Install

### Regular Install

```bash
cd ~/deephyper
pip install .
```

### Developer Install

```note
There currently are 'tensorflow-cpu 2.4.1 requires ...' errors.
We are checking if that breaks the install.
```

```bash
cd ~/deephyper
pip install -e .
```

## Start Ray

Your venv should already be active.  If not,

```bash
source ./venv/bin/activate
```

```bash
(venv) wilsonb@sm-01:~/venvs$ source ~/venvs/dhvenv4/bin/activate
(dhvenv4) wilsonb@sm-01:~/venvs$ ray start --head --node-ip-address=192.168.200.130 --port=6379 --num-cpus 1 --block

```

Leave ray running in this terminal window.

## Open Terminal Window

You must use sm-01.

Run all instructions necessary to get to your correct directory and activate your venv.

## Run

```bash
export XLA_FLAGS=--xla_hlo_profile
deephyper nas random --evaluator ray --ray-address 192.168.200.130:6379 --problem nas_problems.nas_problems.model1.problem.Problem
```

```text
(dhvenv3) wilsonb@sm-01:~/deephyper/model1$ deephyper nas random --evaluator ray --ray-address 192.168.200.130:6379 --problem nas_problems.nas_problems.model1.problem.Problem
2021-08-03 08:39:54.355491: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object file: No such file or directory
2021-08-03 08:39:54.355534: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
 ************************************************************************
   Maximizing the return value of function: deephyper.nas.run.alpha.run
 ************************************************************************
2021-08-03 08:39:57,982	INFO worker.py:727 -- Connecting to existing Ray cluster at address: 192.168.200.130:6379
This search doesn't have an exiting procedure...
(pid=129946) 2021-08-03 08:42:00.312804: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object file: No such file or directory
(pid=129946) 2021-08-03 08:42:00.312862: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.

(pid=129946) 2021-08-03 08:42:03.682756: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcuda.so.1'; dlerror: libcuda.so.1: cannot open shared object file: No such file or directory
(pid=129946) 2021-08-03 08:42:03.682806: W tensorflow/stream_executor/cuda/cuda_driver.cc:326] failed call to cuInit: UNKNOWN ERROR (303)


(pid=129946) 2021-08-03 08:42:03.682841: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:156] kernel driver does not appear to be running on this host (sm-01): /proc/driver/nvidia/version does not exist

(pid=129946) 2021-08-03 08:44:02.738595: I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 FMA

(pid=129946) To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
(pid=129946) 2021-08-03 08:44:04.081314: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:176] None of the MLIR Optimization Passes are enabled (registered 2)

(pid=129946) 2021-08-03 08:44:04.105904: I tensorflow/core/platform/profile_utils/cpu_utils.cc:114] CPU Frequency: 2250100000 Hz

(pid=129946) 2021-08-03 08:44:04.166916: W tensorflow/compiler/jit/mark_for_compilation_pass.cc:1653] (One-time warning): Not using XLA:CPU for cluster because envvar TF_XLA_FLAGS=--tf_xla_cpu_global_jit was not set.  If you want XLA:CPU, either set that envvar, or use experimental_jit_scope to enable XLA:CPU.  To confirm that XLA is active, pass --vmodule=xla_compilation_cache=1 (as a proper command-line flag, not via TF_XLA_FLAGS) or set the envvar XLA_FLAGS=--xla_hlo_profile.



HERE
(pid=129946) /homes/wilsonb/venvs/dhvenv3/lib/python3.7/site-packages/tensorflow/python/keras/utils/generic_utils.py:497: CustomMaskWarning: Custom mask layers require a config and must override get_config. When loading, the custom mask layer must be passed to the custom_objects argument.
(pid=129946)   category=CustomMaskWarning)
```




### Analytics

#### Prepare Jupyter Notebook

replace data_2021-07-14_01.json

```bash
deephyper-analytics parse deephyper.log

Xdeephyper-analytics single -p $MY_JSON_FILE
Xdeephyper notebook --type nas --output mynotebook.ipynb $MY_JSON_FILE

deephyper-analytics notebook --type nas --output dh-analytics-nas.ipynb data_2021-07-14_01.json

tx dh-analytics-nas.ipynb results.csv and init_info* to local machine.

$ jupyter notebook
```

Select dh-analytics-nas.ipynb and run all cells

Output of best models

```bash
'0':
  arch_seq: '[0.8323024001314224, 0.7889290619064494, 0.9385678954207153, 0.16059637997392429,
    0.6488539744120456, 0.8325765404421139, 0.9888735139157468, 0.9322143923769549,
    0.7933123406870071]'
  elapsed_sec: 89.9581792355
  id: 678aa6f4-e42d-11eb-a964-f18f9b2aeba1
  objective: 1.0
'1':
  arch_seq: '[0.11650447341058867, 0.5875885504875177, 0.7616171130499415, 0.40112081475681394,
    0.7493532265120827, 0.2800965658218403, 0.31393139677072335, 0.165157311048022,
    0.8629399531678635]'
  elapsed_sec: 1583.2949447632
  id: 02a893d0-e431-11eb-aa27-f18f9b2aeba1
  objective: 1.0
'2':
  arch_seq: '[0.33164869729598867, 0.20248330614152665, 0.2547469587010932, 0.015167333255724169,
    0.2775206010994029, 0.9916415578893681, 0.4538746929355745, 0.21892060220106446,
    0.9121595786916527]'
  elapsed_sec: 1584.8581268787
  id: 039cd79e-e431-11eb-aa27-f18f9b2aeba1
  objective: 1.0
  ```

Update problem.py by replacing only the "if __name___ ..." section.

```python
# Get model.
if __name__ == '__main__':
    arch_seq = [0.8323024001314224, 0.7889290619064494, 0.9385678954207153, 0.16059637997392429,
    0.6488539744120456, 0.8325765404421139, 0.9888735139157468, 0.9322143923769549,
    0.7933123406870071]
    model = Problem.get_keras_model(arch_seq)

    print('Saving model...')
    model.save('model')

    #
    # Save model config info.
    #

    json_config = model.to_json()
    #new_model = keras.models.model_from_json(json_config)

    import json

    print('Saving model_to_json.json...')
    jsonString = json.dumps(json_config)
    with open("model_to_json.json", "w") as jsonFile:
        jsonFile.write(jsonString)

    config = model.get_config()
    #new_model = keras.Sequential.from_config(config)

    print('Saving model_get_config.json...')
    jsonString = json.dumps(config)
    with open("model_get_config.json", "w") as jsonFile:
        jsonFile.write(jsonString)
```

```bash
git push
```

#### Git Pull From ThetaLogin Node

```bash
ssh wilsonb@theta.alcf.anl.gov

(miniconda-3/latest/base) wilsonb@thetalogin6:~> cd /lus/theta-fs0/projects/datascience/wilsonb/theta/deephyper
git pull
git checkout feature/008-more
```

#### Get on ThetaGPU

```bash
(miniconda-3/latest/base) wilsonb@thetalogin6:~> ssh thetagpusn1

Last login: Wed Jun 30 23:27:50 2021 from thetalogin4.tmi.alcf.anl.gov
wilsonb@thetagpusn1:~$
```

##### Start a node

From thetagpusn1,2

```bash
wilsonb@thetagpusn1:~$ qsub -I -A datascience -t 120 -q full-node -n 1
```

#### On ThetaGPU

##### Run problem.py

```bash
cd /lus/theta-fs0/projects/datascience/wilsonb/theta/deephyper/model1/model1/m1_hps/
source ./SetUpEnv.sh
cd /lus/theta-fs0/projects/datascience/wilsonb/theta/deephyper/
pip3 install -e .

cd nas_problems/nas_problems/model1
python3 problem.py
```

## Keras to PyTorch Conversion

You must do the model conversion with the same TensorFlow and Keras versions with
which the original model was made.

Pip3 will tell you that MMdnn was installed to a directory that is not on the path.

Copy the path that is displayed and update the path.

```bash
pip3 install mmdnn
PATH=/gpfs/mira-home/wilsonb/.local/conda/2021-06-26/bin:$PATH
```

This notes are from [Conversion Reference](https://github.com/fishjump/sketchPytorch).

"Basically, you can follow steps on MMdnn, but I highly recommend you to convert a model step-by-step, don't use mmconvert directly.

1. Convert your model to IR files

```bash
mmtoir  -f keras -iw model.h5 -in model.json -o ir
```

You can get your h5 file by model.save_weights(your_path), and get your json file by model.to_json(). Then, you'll get ir.npy, ir.pd, ir.json.

2. Convert IR files to Pytorch code snippet

```bash
mmtocode -f pytorch -in ir.pb -iw ir.npy -o model.py -ow weight.pkl
```

3. Edit model.py

Because of the compatibility, you may need modify some layers by your self. Please see the output of mmtocode.

\[I don't know why this next line is here.  It looks like a full conversion.\]

```bash
mmconvert -sf keras -iw model.h5 -df tensorflow -om keras_resnet50.dnn
```

"

##### Update Git Repo

```bash
git add -f model
git add -f *.json
git add -f *.h5
git commit -am "New model JSON files."
git push
exit
exit
exit
```

#### On local machine

```bash
git pull
```
## Note

Sam Foreman  7 hours ago
alternatively, it looks like you can use deephyper ray-submit directly from thetagpusn1 to automatically generate and submit a submission script
(documented at the very bottom of https://deephyper.readthedocs.io/en/develop/user_guides/thetagpu.html)
