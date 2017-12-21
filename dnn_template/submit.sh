#!/bin/bash
#PBS -l walltime=5:00:00
#PBS -l nodes=1:ppn=1:gpus=1
#PBS -q gpu

mydev=`cat $PBS_GPUFILE | sed s/.*-gpu// `
export CUDA_VISIBLE_DEVICES=$mydev

cd /remote/gpu04/kasieczka/DeepTop/dnn_template

nice -19  /remote/gpu02/anaconda2/bin/python SimpleTrainHD.py  $MYVAR > output_$MYVAR.txt 2>&1
#nice -19  /remote/gpu02/anaconda2/bin/python SimpleTrainHD.py $NC  2_$MYVAR > output_2_$MYVAR.txt 2>&1
#nice -19  /remote/gpu02/anaconda2/bin/python SimpleTrainHD.py $NC  3_$MYVAR > output_3_$MYVAR.txt 2>&1
#nice -19  /remote/gpu02/anaconda2/bin/python SimpleTrainHD.py $NC  4_$MYVAR > output_4_$MYVAR.txt 2>&1
