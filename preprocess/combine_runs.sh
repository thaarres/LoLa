#!/bin/bash

trap ctrl_c INT
function ctrl_c() {
    exit 1
}

rm -f top+qcd*.h5

nruns=100
for k in $(seq 0 $((nruns-1))); do

    run=$(printf "%03d\n" $k)
    echo $run
    python root2hdf5_michael.py sigruns/$run/tops_run_$run.root bkgruns/$run/qcd_run_$run.root 100 &> /dev/null

done
