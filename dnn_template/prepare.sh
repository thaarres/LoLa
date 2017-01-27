cd /scratch/snx3000/gregork/outputs
SAMPLE=NNXd_et_5deg_sample_v8_v31_v31
for d in $(ls $SAMPLE_*/*.yaml); do  cp $d $(dirname $d).yaml; done; 
for d in $(ls $SAMPLE_*/*.hdf5); do  cp $d $(dirname $d)_weights_latest.hdf5; done; 
for d in $(ls $SAMPLE_*/loss_latest.png); do  cp $d $(dirname $d)_loss_latest.png; done; cp *.png ~
cd -
