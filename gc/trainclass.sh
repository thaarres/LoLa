#!/bin/bash

# Anaconda
export PATH=/mnt/t3nfs01/data01/shome/gregor/anaconda2/bin:$PATH

# Get us ROOT
. /afs/cern.ch/sw/lcg/external/gcc/4.9/x86_64-slc6-gcc49-opt/setup.sh
. /afs/cern.ch/sw/lcg/app/releases/ROOT/6.06.04/x86_64-slc6-gcc49-opt/root/bin/thisroot.sh


# Only use one core
export OMP_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MKL_NUM_THREADS=1

# avoid lock-issues 
export THEANO_FLAGS="base_compiledir=.,gcc.cxxflags=-march=core2"

#print out the environment
env
set -e

pwd
ls -al

#go to work directory
cd $MY_SCRATCH

echo "Run TrainClassifiers.py"
python /mnt/t3nfs01/data01/shome/gregor/DeepTop/dnn_template/TrainClassifiers.py
echo "Done TrainClassifiers.py"



