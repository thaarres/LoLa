module load daint-gpu
module load craype-accel-nvidia60
module load pycuda

/usr/sbin/mongod --dbpath . --port 23888  --directoryperdb --journal --nohttpinterface


