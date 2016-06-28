#!/bin/bash

echo "Go to scratch"
cd $SCRATCH

echo "PWD:"
pwd

echo "Prepare modules"
module swap PrgEnv-cray/5.2.82 PrgEnv-gnu/5.2.40
module load Python/2.7.10-CrayGNU-5.2.40
module load pycuda/2015.1-CrayGNU-5.2.40-Python-2.7.10
module load h5py/2.5.0-CrayGNU-5.2.40-Python-2.7.10-serial
module swap cudatoolkit/6.5.14-1.0502.9613.6.1  cudatoolkit/7.0.28-1.0502.10742.5.1
module load matplotlib/1.4.3-CrayGNU-5.2.40-Python-2.7.10

# Enable CUDNN
export CUDNN_BASE=$SCRATCH/cuda
export LD_LIBRARY_PATH=$CUDNN_BASE/lib64:$LD_LIBRARY_PATH
export CPATH=$CUDNN_BASE/include:$CPATH
export LIBRARY_PATH=$CUDNN_BASE/lib64:$LD_LIBRARY_PATH

# avoid lock-issues 
export THEANO_FLAGS="base_compiledir=$SCRATCH/theano.NOBACKUP"

cd DeepTop/dnn_template

echo "Starting TrainClassifiers.py"
python TrainClassifiers.py

echo "Done.."
