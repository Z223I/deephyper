#!/bin/bash
#COBALT -n 1
#COBALT -t 12:00:00 -q full-node
#COBALT -A APSPolarisI2E
#COBALT --attrs=pubnet
#COBALT -o /lus/theta-fs0/projects/Deep_WF/YYD/Results/COBALT/

#submisstion script for running

echo "Running Cobalt Job $COBALT_JOBID."

echo "Start Ray Cluster"
cd /lus/theta-fs0/projects/datascience/wilsonb/theta/deephyper/model1/model1/m1_hps/
./SingleNodeRayCluster.sh

echo "Start Conda Env"
source ./SetUpEnv.sh

echo "Which python?"
which python

echo "Run DeepHyper"
cd /lus/theta-fs0/projects/datascience/wilsonb/theta/deephyper
deephyper nas random --evaluator ray --ray-address auto --problem nas_problems.nas_problems.model1.problem.Problem --num-cpus-per-task 1 --num-gpus-per-task 1

