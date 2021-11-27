# Quickstart NAS Model 1

## ssh to Login Node

```bash
theta
```

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
qsub -I -A $PROJECT_NAME -n 1 -t 60 -q [single-gpu | full-node]
```

## ThetaGPU

### Start Ray Cluster

```bash
cd /lus/theta-fs0/projects/datascience/wilsonb/theta/deephyper/nas_problems/nas_problems/model1
./SingleNodeRayCluster.sh
```

### Start Conda Env

```bash
source ./SetUpEnv.sh
```

### Run DeepHyper

```bash
cd /lus/theta-fs0/projects/datascience/wilsonb/theta/deephyper
```

If you have your own fork of DeepHyper, do a developer install only the first time.

```bash
pip3 install -e .
```

```bash
cd /lus/theta-fs0/projects/datascience/wilsonb/theta/deephyper/nas_problems/nas_problems/model1/
deephyper nas random --evaluator ray --ray-address auto --problem nas_problems.nas_problems.model1.problem.Problem --num-cpus-per-task 1 --num-gpus-per-task 1
cd /lus/theta-fs0/projects/datascience/wilsonb/theta/deephyper/nas_problems/nas_problems/model1/
deephyper nas random --evaluator ray --ray-address auto --problem nas_problems.nas_problems.model1.problem.Problem --num-cpus-per-task 8 --num-gpus-per-task 8
```







### Analytics

#### Prepare Jupyter Notebook

```bash
deephyper-analytics parse deephyper.log

deephyper-analytics notebook --type nas --output dh-analytics-nas data_2021-11-25_16.json

tx dh-analytics-nas.ipynb results.csv and init_info* to local machine.
```

Change

```python
path_to_logdir = 'data_2021-11-25_16.json'
```

to

```python
path_to_logdir = '.'
```

Change

```python
init_infos_path = os.path.join(path_to_logdir, "init_infos.json")
```

to

```python
init_infos_path = os.path.join(path_to_logdir, "dh-analytics-nas")
```




```bash
pip install jupyterlab

jupyter notebook
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

When that finishes...

```bash
git commit -am "Keras model saved.""
git push
exit
exit
```

## Hey!  Diff 009 with 011

model.json
model/*
model.h5

```text
From one of our model engineers: I believe this should work, we used the torch save/load_state_dict in the past for hermit as well. As long as the parameter values on host are correct, i.e. if training we called model.cpu() after finishing, the parameters saved should be correct. And for loading it should probably be done before the samba.from_torch_(model), though it might work after (not sure).
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
