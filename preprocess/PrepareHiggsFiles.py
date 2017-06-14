import os
import glob
import time

import numpy as np
import pandas

import pdb

paths_to_process = ["/scratch/gregor/qcd/",
                    "/scratch/gregor/wz/",
                    "/scratch/gregor/zh/",
                    "/scratch/gregor/zhb/",
                    "/scratch/gregor/zz/"]

for path in paths_to_process:
    input_files = glob.glob( os.path.join(path,"*.h5"))

    for ifile, infile_name in enumerate(input_files):
        
        store = pandas.HDFStore(infile_name)
        
        open_time = time.time()

        df_in = store.select("table")

        pdb.set_trace()

#        # Train / Test / Validate
#        # ttv==0: 60% Train
#        # ttv==1: 20% Test
#        # ttv==2: 20% Final Validation
#        df_in["ttv"] = np.random.choice([0,1,2], df_in.shape[0], p=[0.6, 0.2, 0.2])
#
#        train = df_in[ df_in["ttv"]==0]
#        test  = df_in[ df_in["ttv"]==1]
#        val   = df_in[ df_in["ttv"]==2]
#
#        train.to_hdf('deeph-train-v1.h5','table',append=True, complib = "blosc", complevel=5)
#        test.to_hdf('deeph-test-v1.h5','table',append=True, complib = "blosc", complevel=5)
#        val.to_hdf('deeph-val-v1.h5','table',append=True, complib = "blosc", complevel=5)
#
#        close_time = time.time()
#
#        print "Sample: {0}, File: {1}/{2}, Time for file: {3}s, TTV: {4}/{5}/{6}".format(
#            path, 
#            ifile, len(input_files),
#            (close_time - open_time),
#            len(train), len(test), len(val)
#        )
#
