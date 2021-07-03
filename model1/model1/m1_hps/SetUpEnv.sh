#!/bin/bash

. /etc/profile

# This module was created by Kyle.
module load conda/deephyper/0.2.5


PATH_TO_ENV="/enter/path"

# Activate conda env
conda activate

export PYTHONPATH=$HOME/deephyper_pytorch_layers:$PYTHONPATH