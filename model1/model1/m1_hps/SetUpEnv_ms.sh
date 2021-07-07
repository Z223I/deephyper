#!/bin/bash

. /etc/profile

# Tensorflow optimized for A100 with CUDA 11
module load conda/deephyper/0.2.5

# Activate conda env
conda activate

export PYTHONPATH=$HOME/deephyper_pytorch_layers:$PYTHONPATH
