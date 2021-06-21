Directory structure
===================
```
benchmarks
    directory for problems
experiments
    directory for saving the running the experiments and storing the results
search
    directory for source files
```
Install instructions
====================

With anaconda do the following:

```
conda create -n dl-hps python=3
conda activate dl-hps
conda install h5py
conda install scikit-learn
conda install pandas
conda install mpi4py
conda install -c conda-forge keras
conda install -c conda-forge scikit-optimize
git clone https://github.com/scikit-optimize/scikit-optimize.git
cd scikit-optimize
pip install -e.
conda install -c conda-forge xgboost
```

Or, use a shell script:

```
conda create -n dl-hps python=3
conda activate dl-hps
conda install -y h5py
conda install -y scikit-learn
conda install -y pandas
conda install -y mpi4py
conda install -y -c conda-forge keras
conda install -y -c conda-forge scikit-optimize
git clone https://github.com/scikit-optimize/scikit-optimize.git
cd scikit-optimize
pip install -e .
conda install -y -c conda-forge xgboost
```


Usage (with Balsam)
=====================


Run once
----------
```
    source activate dl-hps   # balsam is installed here too (commands like “balsam ls” must work)

    cd directory_containing_dl-hps
    mv dl-hps dl_hps         # important: change to underscore (import system relies on this)

    cd dl_hps/search
    balsam app --name search --description "run async_search" --executable async-search.py
```

From a qsub bash script (or in this case, an interactive session)
----------------------------------------------------------------------
```
    qsub -A datascience -n 8 -t 60 -q debug-cache-quad -I

    source ~/.bash_profile    # this should set LD_library_path correctly for mpi4py and make conda available (see balsam quickstart guide)
    source activate dl-hps   # balsam is installed here too (commands like “balsam ls” should work)

    balsam job --name test --workflow b1_addition --app search --wall-minutes 20 --num-nodes 1 --ranks-per-node 1 --args '--max_evals=20'

    balsam launcher --consume --max-ranks-per-node 4
    # will auto-recognize the nodes and allow only 4 addition_rnn.py tasks to run simultaneously on a node
```

To restart:
----------------------------------------------------------------------
If async-search.py stops for any reason, it will create checkpoint files in the
search working directory.  If you simply restart the balsam launcher, it will
resume any timed-out jobs which have the state "RUN_TIMEOUT".  The async-search
will automatically resume where it left off by finding the checkpoint files in
its working directory.

Alternatively, async-search may have completed, but you wish to extend the
optimization with more iterations.  In this case, you can create a new
async-search job and specify the argument "--restart-from" with the full path
to the previous run's working directory.

```
    # To simply re-start timed-out jobs:
    balsam launcher --consume --max-ranks 4

    # To create a new job extending a previously finished optimization
    balsam job --name test --workflow b1_addition --app search --wall-minutes 20 --num-nodes 1 --ranks-per-node 1 --args '--max_evals=20 --restart-from    /path/to/previous/search/directory'
    balsam launcher --consume --max-ranks-per-node 4
```
