#!/bin/bash

echo "Go to scratch"
cd $SCRATCH

echo "PWD:"
pwd

echo "Prepare modules"
module load daint-gpu
module load craype-accel-nvidia60
module swap cudatoolkit/8.0.54_2.2.8_ga620558-2.1 cudatoolkit/8.0.44_GA_2.2.7_g4a6c213-2.1
module load pycuda/2016.1.2-CrayGNU-2016.11-Python-3.5.2-cuda-8.0


export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH

# Enable CUDNN
export CUDNN_BASE=/users/gregork/cuda
export LD_LIBRARY_PATH=$CUDNN_BASE/lib64:$LD_LIBRARY_PATH
export CPATH=$CUDNN_BASE/include:$CPATH
export LIBRARY_PATH=$CUDNN_BASE/lib64:$LD_LIBRARY_PATH

# avoid lock-issues 
export THEANO_FLAGS="base_compiledir=$SCRATCH/theano.NOBACKUP.$(date +%s).$RANDOM"

echo "Go back home"
cd ~/DeepTop/dnn_template

echo "Preparing tunnel"
ssh -fL 9000:daint101:23836 daint101 sleep 84000

export PYTHONPATH=/users/gregork/DeepTop/dnn_template:$PYTHONPATH

echo "Starting worker"
 ~/.local/bin/hyperopt-mongo-worker --mongo=localhost:9000/foo_db --poll-interval=10 --max-consecutive-failures=2000

echo "Done.."
