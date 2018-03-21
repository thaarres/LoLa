#!/usr/bin/env python

import glob
import numpy as np
import pandas
import sys
import h5py



#say sig_branches etc. have comma-separated columns in format:
#E0 E1...En_cands px0...pxn_cands py0...pyn_cands pz0...pzn_cands
n_cands = 10
version = "v17_{0}nc".format(n_cands)

sigfile, bkgfile = 'sig.csv', 'bkg.csv'
df_sig = pandas.read_csv(sigfile,header=0)
df_bkg = pandas.read_csv(bkgfile,header=0)

df_sig["is_signal"] = 1
df_bkg["is_signal"] = 0
df = pandas.concat([df_sig, df_bkg], ignore_index=True)
   
df = df.iloc[np.random.permutation(len(df))]

# Train / Test / Validate
# ttv==0: 60% Train
# ttv==1: 20% Test
# ttv==2: 20% Final Validation
df["ttv"] = np.random.choice([0,1,2], df.shape[0],p=[0.6, 0.2, 0.2])

train = df[ df["ttv"]==0 ]
test  = df[ df["ttv"]==1 ]
val   = df[ df["ttv"]==2 ]
        

train.to_hdf('top+qcdconst-train-{0}.h5'.format(version),'table',append=True)
test.to_hdf('top+qcdconst-test-{0}.h5'.format(version),'table',append=True)
val.to_hdf('top+qcdconst-val-{0}.h5'.format(version),'table',append=True)
