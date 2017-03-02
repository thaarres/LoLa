module load daint-gpu
module load craype-accel-nvidia60
module swap cudatoolkit/8.0.54_2.2.8_ga620558-2.1 cudatoolkit/8.0.44_GA_2.2.7_g4a6c213-2.1
module load pycuda/2016.1.2-CrayGNU-2016.11-Python-3.5.2-cuda-8.0

/usr/sbin/mongod --dbpath . --port 23836  --directoryperdb --journal --nohttpinterface


