#######################################
# Imports
########################################

print "Imports: Starting..."

import sys
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


########################################
# Configuration
########################################

n_cands = 20
batch_size = 2000

cols  = ["E_{0}".format(i_cand) for i_cand in range(n_cands)] 
cols += ["X_{0}".format(i_cand) for i_cand in range(n_cands)]
cols += ["Y_{0}".format(i_cand) for i_cand in range(n_cands)]
cols += ["Z_{0}".format(i_cand) for i_cand in range(n_cands)]
cols += ["is_signal_new"]

for is_signal, infname in [[1, "fatjets_4momenta_sig_large.root"],
                           [0, "fatjets_4momenta_bkg_large.root"]]:
                           
    print infname

    peek = root_numpy.root2array(infname, treename="tree", branches = ["htt_tagged"])
    n_batches =  len(peek)/batch_size - 1

    last_time = time.time()

    for ibatch in range(n_batches):

        open_time = time.time()

        print ibatch, "/", n_batches

        batch_array = root_numpy.root2array(infname, treename="tree", start=ibatch * batch_size, stop = (ibatch+1)*batch_size)

        df = pandas.DataFrame(batch_array['tau2'],columns=["tau2"])

        ve  = np.array([x for x in batch_array["ve"]])
        vpx = np.array([x for x in batch_array["vpx"]])
        vpy = np.array([x for x in batch_array["vpy"]])
        vpz = np.array([x for x in batch_array["vpz"]])

        for i in range(n_cands):            
            df["E_{0}".format(i)]  = ve[:,i]
            df["PX_{0}".format(i)] = vpx[:,i]
            df["PY_{0}".format(i)] = vpy[:,i]
            df["PZ_{0}".format(i)] = vpz[:,i]
        
        # Train / Test / Validate
        # ttv==0: 60% Train
        # ttv==1: 20% Test
        # ttv==2: 20% Final Validation
        df["ttv"] = np.random.choice([0,1,2], df.shape[0],p=[0.6, 0.2, 0.2])

        df["is_signal_new"] = is_signal
            
        train = df[ df["ttv"]==0 ]
        test  = df[ df["ttv"]==1 ]
        val   = df[ df["ttv"]==2 ]
        
        print len(df), len(train), len(test), len(val)

        train.to_hdf('topconst-train-v4.h5','table',append=True)
        test.to_hdf('topconst-test-v4.h5','table',append=True)
        val.to_hdf('topconst-val-v4.h5','table',append=True)

        close_time = time.time()

        print "Time per 1k events: ", 1000 * (close_time - open_time)/batch_size 



