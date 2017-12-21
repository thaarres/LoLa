#!/bin/bash
#PBS -l walltime=10:00:00
#PBS -l nodes=1:ppn=1:gpus=1
#PBS -q gpu

mydev=`cat $PBS_GPUFILE | sed s/.*-gpu// `
export CUDA_VISIBLE_DEVICES=$mydev

cd /remote/gpu04/kasieczka/DeepTop/dnn_template

nice -19  /remote/gpu02/anaconda2/bin/python SimpleTrainHD-Higgs.py $NC  1_$MYVAR
