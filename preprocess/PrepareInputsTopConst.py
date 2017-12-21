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

n_cands = int(sys.argv[1])
batch_size = 1000
do_charge = False
version = "v17_{0}nc".format(n_cands)

cols  = ["E_{0}".format(i_cand) for i_cand in range(n_cands)] 
cols += ["X_{0}".format(i_cand) for i_cand in range(n_cands)]
cols += ["Y_{0}".format(i_cand) for i_cand in range(n_cands)]
cols += ["Z_{0}".format(i_cand) for i_cand in range(n_cands)]
if do_charge:
    cols += ["C_{0}".format(i_cand) for i_cand in range(n_cands)]
cols += ["is_signal_new"]

for is_signal, infname in [[1, "350tt_b.root"],
                           [0, "350qcd_b.root"]]:

#for is_signal, infname in [[1, "antikT_sig_boosted.root"],
#                           [0, "antikT_bkg_boosted.root"]]:
                           
    print infname

    peek = root_numpy.root2array(infname, treename="tree", branches = ["ve"])
    n_batches =  270 #len(peek)/batch_size - 1

    last_time = time.time()

    for ibatch in range(n_batches):

        open_time = time.time()

        print ibatch, "/", n_batches

        batch_array = root_numpy.root2array(infname, treename="tree", start=ibatch * batch_size, stop = (ibatch+1)*batch_size)

        df = pandas.DataFrame()

        ve  = np.array([x for x in batch_array["ve"]])
        vpx = np.array([x for x in batch_array["vpx"]])
        vpy = np.array([x for x in batch_array["vpy"]])
        vpz = np.array([x for x in batch_array["vpz"]])
        if do_charge:
            vc  = np.array([x for x in batch_array["vc"]])
        
    

        try:
            if len(vpx.shape)==2:
               
                pass
            else:
                print "Skipping:"
                continue

            if vpx.shape[1]==100:
                
                pass
            else:
                print "Skipping:"
                continue
        except:
            print "Skipping:"
            continue
        

        idx = (-1*(pow(vpx,2) + pow(vpy,2))).argsort()

        dims0 = np.expand_dims(np.arange(batch_size),-1)
        dims1 = np.arange(batch_size)

        
#        print idx.shape
#        print idx
#        print idx[272]

        ve = ve[dims0,idx[dims1]]
        vpx = vpx[dims0,idx[dims1]]
        vpy = vpy[dims0,idx[dims1]]
        vpz = vpz[dims0,idx[dims1]]
        if do_charge:
            vc = vc[dims0,idx[dims1]]




        for i in range(n_cands):            
            df["E_{0}".format(i)]  = ve[:,i]
            df["PX_{0}".format(i)] = vpx[:,i]
            df["PY_{0}".format(i)] = vpy[:,i]
            df["PZ_{0}".format(i)] = vpz[:,i]
            if do_charge:
                df["C_{0}".format(i)]  = vc[:,i]
        
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

        train.to_hdf('topconst-train-{0}.h5'.format(version),'table',append=True)
        test.to_hdf('topconst-test-{0}.h5'.format(version),'table',append=True)
        val.to_hdf('topconst-val-{0}.h5'.format(version),'table',append=True)

        close_time = time.time()

        print "Time per 1k events: ", 1000 * (close_time - open_time)/batch_size 



