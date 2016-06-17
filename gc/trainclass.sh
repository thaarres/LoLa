#!/bin/bash

# Anaconda
export PATH=/mnt/t3nfs01/data01/shome/gregor/anaconda2/bin:$PATH

export TTH_STAGEOUT_PATH=/home/gregor/tth/gc

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

echo "Run TrainClassifiers.py"
python /mnt/t3nfs01/data01/shome/gregor/DeepTop/dnn_template/TrainClassifiers.py
echo "Done TrainClassifiers.py"

#copy output
OUTDIR=$HOME/deeptop/dnn_test_10/${TASK_ID}/${MY_JOBID}/
mkdir -p $OUTDIR 
echo "copying output"

cp -v *.png $OUTDIR 
cp -v valacc.txt $OUTDIR 
cp -v maxvalacc.txt $OUTDIR 
cp -v deltaacc.txt $OUTDIR 
cp -v *.yaml $OUTDIR 
cp -v *.h5 $OUTDIR 
env > $OUTDIR/env.txt



