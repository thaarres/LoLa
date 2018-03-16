#######################################
# Imports
########################################

print "Imports: Starting..."

import sys
if len(sys.argv) != 4:
    print "Enter signal_file background_file n_constituents as arguments"
    sys.exit()
    
import os
import pickle
import pdb

print "Imported basics"

import ROOT
print "Imported ROOT"

import numpy as np
import pandas
import root_numpy
import h5py

import time

print "Imports: Done..."

##doesn't work if input files large - too much memory

########################################
# Configuration
########################################


infname_sig = sys.argv[1]
infname_bkg = sys.argv[2]
n_cands = int(sys.argv[3])
version = "v17_{0}nc".format(n_cands)


## load data
start = time.time()
df_sig = pandas.DataFrame(root_numpy.root2array(infname_sig, treename="tree"))
df_bkg = pandas.DataFrame(root_numpy.root2array(infname_bkg, treename="tree"))
df_all = pandas.concat([df_sig, df_bkg], ignore_index=True)
print time.time()-start, "seconds to load data"


## shuffle data
start = time.time()
df_all = df_all.iloc[np.random.permutation(len(df_all))].reset_index(drop=True)
print time.time()-start, "seconds to shuffle data"
    
open_time = time.time()

df = pandas.DataFrame()

df["truthE"] = df_all["truth_e"]
df["truthPX"] = df_all["truth_px"]
df["truthPY"] = df_all["truth_py"]
df["truthPZ"] = df_all["truth_pz"]
df["is_signal_new"] = df_all["is_signal"]
for i in range(n_cands):            
    df["E_{0}".format(i)] = [x[i] for x in df_all["e"]]
    df["PX_{0}".format(i)] = [x[i] for x in df_all["px"]]
    df["PY_{0}".format(i)] = [x[i] for x in df_all["py"]]
    df["PZ_{0}".format(i)] = [x[i] for x in df_all["pz"]]

# Train / Test / Validate
# ttv==0: 60% Train
# ttv==1: 20% Test
# ttv==2: 20% Final Validation
df["ttv"] = np.random.choice([0,1,2], df.shape[0],p=[0.6, 0.2, 0.2])

train = df[ df["ttv"]==0 ]
test  = df[ df["ttv"]==1 ]
val   = df[ df["ttv"]==2 ]
        
print len(df), len(train), len(test), len(val)

train.to_hdf('top+qcdconst-train-{0}.h5'.format(version),'table',append=True)
test.to_hdf('top+qcdconst-test-{0}.h5'.format(version),'table',append=True)
val.to_hdf('top+qcdconst-val-{0}.h5'.format(version),'table',append=True)

close_time = time.time()

print "Time for the lot: ", (close_time - open_time)
