#!/bin/bash

. /etc/profile

# Tensorflow optimized for A100 with CUDA 11
module load conda/2021-06-26

# Activate conda env
conda activate

# This will allow you to run DeepHyper from anywhere.
PATH_TO_ENV=/lus/theta-fs0/projects/datascience/wilsonb/theta/deephyper
export PYTHONPATH=$PATH_TO_ENV:$PYTHONPATH
