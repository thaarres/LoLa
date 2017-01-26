cd /scratch/snx3000/gregork/outputs
SAMPLE=NNXd_min_5deg_sample_v7_v27_v27
for d in $(ls $SAMPLE_*/*.yaml); do  cp $d $(dirname $d).yaml; done; 
for d in $(ls $SAMPLE_*/*.hdf5); do  cp $d $(dirname $d)_weights_latest.hdf5; done; 
for d in $(ls $SAMPLE_*/loss_latest.png); do  cp $d $(dirname $d)_loss_latest.png; done; cp *.png ~
cd -
