# DeepHyper NAS Running Local

## Python Environment

This work is being done with Python 3.8.7.  It might be possible to use a newer Python version.

### Create Python 3.8.7 Venv

Do this in your preferred location.

```bash
cd /path/to/venvs/directory
python3 -m venv /path/to/new/virtual/environment
source /path/to/new/virtual/environment/bin/activate
```

### Update Venv

#### Requirements.txt

Note:  You will find a 'developer' install of DeepHyper in the following file.  You will need to update that per your situation.

```text
absl-py==0.13.0
aiohttp==3.7.4.post0
aiohttp-cors==0.7.0
aioredis==1.3.1
astunparse==1.6.3
async-timeout==3.0.1
attrs==21.2.0
blessings==1.7
cachetools==4.2.2
certifi==2021.5.30
chardet==4.0.0
charset-normalizer==2.0.3
click==8.0.1
cloudpickle==1.6.0
colorama==0.4.4
colorful==0.5.4
ConfigSpace==0.4.18
cycler==0.10.0
Cython==0.29.24
deap==1.3.1
decorator==4.4.2
-e git+https://github.com/z223i/deephyper.git@3b4c8b992896c25e2500f13c8138d84aead1a34c#egg=deephyper
deepspace==0.0.4
dh-scikit-optimize==0.8.3
dm-tree==0.1.6
filelock==3.0.12
flatbuffers==1.12
gast==0.4.0
google-api-core==1.31.0
google-auth==1.33.0
google-auth-oauthlib==0.4.4
google-pasta==0.2.0
googleapis-common-protos==1.53.0
gpustat==0.6.0
grpcio==1.34.1
h5py==3.1.0
hiredis==2.0.0
idna==3.2
Jinja2==3.0.1
joblib==1.0.1
jsonschema==3.2.0
Keras==2.4.3
keras-nightly==2.5.0.dev2021032900
Keras-Preprocessing==1.1.2
kiwisolver==1.3.1
liac-arff==2.5.0
Markdown==3.3.4
MarkupSafe==2.0.1
matplotlib==3.4.2
mmdnn==0.3.1
msgpack==1.0.2
multidict==5.1.0
networkx==2.5.1
numpy==1.19.5
nvidia-ml-py3==7.352.0
oauthlib==3.1.1
opencensus==0.7.13
opencensus-context==0.1.2
openml==0.10.2
opt-einsum==3.3.0
packaging==21.0
pandas==1.3.0
Pillow==8.3.1
prometheus-client==0.11.0
protobuf==3.17.3
psutil==5.8.0
py-spy==0.3.7
pyaml==20.4.0
pyasn1==0.4.8
pyasn1-modules==0.2.8
pydantic==1.8.2
pydot==1.4.2
pyparsing==2.4.7
pyrsistent==0.18.0
python-dateutil==2.8.2
pytz==2021.1
PyYAML==5.4.1
ray==1.4.1
redis==3.5.3
requests==2.26.0
requests-oauthlib==1.3.0
rsa==4.7.2
scikit-learn==0.24.2
scipy==1.7.0
six==1.15.0
tensorboard==2.5.0
tensorboard-data-server==0.6.1
tensorboard-plugin-wit==1.8.0
tensorflow==2.5.0
tensorflow-estimator==2.5.0
tensorflow-probability==0.13.0
termcolor==1.1.0
threadpoolctl==2.2.0
tqdm==4.61.2
typeguard==2.12.1
typing-extensions==3.7.4.3
urllib3==1.26.6
Werkzeug==2.0.1
wrapt==1.12.1
xgboost==1.4.2
xmltodict==0.12.0
yarl==1.6.3
```

```bash
pip3 install -r requirements.txt
```

## Run DeepHyper

### Start DeepHyper

```bash
deephyper nas random --evaluator deephyper.evaluator.SubProcessEvaluator --problem nas_problems.nas_problems.model1.problem.Problem --max-evals 10 --num-cpus-per-task 6

deephyper nas random --evaluator deephyper.evaluator.RayEvaluator --problem nas_problems.nas_problems.model1.problem.Problem --max-evals 10 --num-cpus-per-task 6
```