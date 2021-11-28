#!/bin/bash

# Necessary for Bash shells
. /etc/profile

# Tensorflow optimized for A100 with CUDA 11
module load conda/2021-09-22

# Activate conda env
conda activate $CONDA_ENV_PATH

# Activate XLA optimized compilation
export TF_XLA_FLAGS=--tf_xla_enable_xla_devices