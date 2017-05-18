#!/bin/bash

echo "hostname"
hostname

echo "Go to scratch"
cd $SCRATCH

echo "PWD:"
pwd



echo "Prepare modules"
module load daint-gpu
module load cudatoolkit
module load pycuda


export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH

# Enable CUDNN
export CUDNN_BASE=/users/gregork/cuda
export LD_LIBRARY_PATH=$CUDNN_BASE/lib64:$LD_LIBRARY_PATH
export CPATH=$CUDNN_BASE/include:$CPATH
export LIBRARY_PATH=$CUDNN_BASE/lib64:$LD_LIBRARY_PATH

# avoid lock-issues 
export THEANO_FLAGS="base_compiledir=$SCRATCH/theano.NOBACKUP.$(date +%s).$RANDOM"

export OMP_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MKL_NUM_THREADS=1


echo "Go back home"
cd ~/DeepW/dnn_template

echo "Preparing tunnel"
ssh -fL 9000:daint101:23836 daint101 sleep 84000

echo "lsof"
lsof -i -n | egrep '\<ssh\>'

export PYTHONPATH=/users/gregork/DeepW/dnn_template:$PYTHONPATH

echo "python version:"
python --version

echo "Starting worker"

 ~/.local/bin/hyperopt-mongo-worker --mongo=localhost:9000/foo_db --poll-interval=10 --max-consecutive-failures=2000

echo "Done.."
