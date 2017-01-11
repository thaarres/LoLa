#!/bin/bash

echo "Go to scratch"
cd $SCRATCH

echo "PWD:"
pwd

echo "Prepare modules"
#module swap PrgEnv-cray/5.2.82 PrgEnv-gnu/5.2.40
#module load Python/2.7.10-CrayGNU-5.2.40
#module load pycuda/2015.1-CrayGNU-5.2.40-Python-2.7.10
#module load h5py/2.5.0-CrayGNU-5.2.40-Python-2.7.10-serial
#module swap cudatoolkit/6.5.14-1.0502.9613.6.1  cudatoolkit/7.0.28-1.0502.10742.5.1
#module load matplotlib/1.4.3-CrayGNU-5.2.40-Python-2.7.10

module load daint-gpu
module load craype-accel-nvidia60
module load pycuda/2016.1.2-CrayGNU-2016.11-Python-3.5.2-cuda-8.0

# Enable CUDNN
export CUDNN_BASE=/users/gregork/cuda
export LD_LIBRARY_PATH=$CUDNN_BASE/lib64:$LD_LIBRARY_PATH
export CPATH=$CUDNN_BASE/include:$CPATH
export LIBRARY_PATH=$CUDNN_BASE/lib64:$LD_LIBRARY_PATH


# avoid lock-issues 
export THEANO_FLAGS="base_compiledir=$SCRATCH/theano.NOBACKUP.$(date +%s)"



echo "Go back home"
cd ~/DeepTop/dnn_template

echo "Received" $1

echo "Starting TrainClassifiers.py"
python TrainClassifiers.py $1 &> $1_log.txt

echo "Done.."
