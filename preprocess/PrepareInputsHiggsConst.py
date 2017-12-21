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
do_charge = True
do_vertex = True
version = "v17b_cand{0}_c{1}_vtx{2}".format(n_cands, do_charge, do_vertex)

print "version=",version

cols  = ["E_{0}".format(i_cand) for i_cand in range(n_cands)] 
cols += ["X_{0}".format(i_cand) for i_cand in range(n_cands)]
cols += ["Y_{0}".format(i_cand) for i_cand in range(n_cands)]
cols += ["Z_{0}".format(i_cand) for i_cand in range(n_cands)]

if do_charge:
    cols += ["C_{0}".format(i_cand) for i_cand in range(n_cands)]
if do_vertex:
    cols  = ["VX_{0}".format(i_cand) for i_cand in range(n_cands)] 
    cols  = ["VY_{0}".format(i_cand) for i_cand in range(n_cands)] 
    cols  = ["VZ_{0}".format(i_cand) for i_cand in range(n_cands)] 

cols += ["is_signal_new"]

# TOP
for is_signal, infname in [[0, "350qcd_b.root"],
                           [1, "350tt_b.root"]]:

# HIGGS
#for is_signal, infname in [[0, "/scratch/gregor/higgs/qcd.root"],
#                           [1, "/scratch/gregor/higgs/wz.root" ],
#                           [2, "/scratch/gregor/higgs/zhb.root"], 
#                           [3, "/scratch/gregor/higgs/zh.root" ],
#                           [4, "/scratch/gregor/higgs/zz.root" ]]:
                           
    print infname

    peek = root_numpy.root2array(infname, treename="tree", branches = ["ve"])
    n_batches =  len(peek)/batch_size - 1

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
        if do_vertex:
            vvertX = np.array([x for x in batch_array["vvertX"]])
            vvertY = np.array([x for x in batch_array["vvertY"]])
            vvertZ = np.array([x for x in batch_array["vvertZ"]])
                    
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

        ve = ve[dims0,idx[dims1]]
        vpx = vpx[dims0,idx[dims1]]
        vpy = vpy[dims0,idx[dims1]]
        vpz = vpz[dims0,idx[dims1]]
        if do_charge:
            vc = vc[dims0,idx[dims1]]
        if do_vertex:            
            vvertX = vvertX[dims0,idx[dims1]]
            vvertY = vvertY[dims0,idx[dims1]]
            vvertZ = vvertZ[dims0,idx[dims1]]

        charged_ve     = np.empty((batch_size,20))
        charged_vpx    = np.empty((batch_size,20))
        charged_vpy    = np.empty((batch_size,20))
        charged_vpz    = np.empty((batch_size,20))
        charged_vc     = np.empty((batch_size,20))
        charged_vvertX = np.empty((batch_size,20))
        charged_vvertY = np.empty((batch_size,20))
        charged_vvertZ = np.empty((batch_size,20))

        notcharged_ve     = np.empty((batch_size,20))
        notcharged_vpx    = np.empty((batch_size,20))
        notcharged_vpy    = np.empty((batch_size,20))
        notcharged_vpz    = np.empty((batch_size,20))
        notcharged_vc     = np.empty((batch_size,20))
        notcharged_vvertX = np.empty((batch_size,20))
        notcharged_vvertY = np.empty((batch_size,20))
        notcharged_vvertZ = np.empty((batch_size,20))

        for irow in np.arange(batch_size):
            mask = (vc[irow]*vc[irow] == 1)
            antimask = np.logical_not(mask)

            charged_ve[irow]     = np.resize(ve[irow,mask],(20))
            charged_vpx[irow]    = np.resize(vpx[irow,mask],(20))
            charged_vpy[irow]    = np.resize(vpy[irow,mask],(20))
            charged_vpz[irow]    = np.resize(vpz[irow,mask],(20))
            charged_vc[irow]     = np.resize(vc[irow,mask],(20))
            charged_vvertX[irow] = np.resize(vvertX[irow,mask],(20))
            charged_vvertY[irow] = np.resize(vvertY[irow,mask],(20))
            charged_vvertZ[irow] = np.resize(vvertZ[irow,mask],(20))    

            notcharged_ve[irow]     = np.resize(ve[irow,antimask],(20))
            notcharged_vpx[irow]    = np.resize(vpx[irow,antimask],(20))
            notcharged_vpy[irow]    = np.resize(vpy[irow,antimask],(20))
            notcharged_vpz[irow]    = np.resize(vpz[irow,antimask],(20))
            notcharged_vc[irow]     = np.resize(vc[irow,antimask],(20))
            notcharged_vvertX[irow] = np.resize(vvertX[irow,antimask],(20))
            notcharged_vvertY[irow] = np.resize(vvertY[irow,antimask],(20))
            notcharged_vvertZ[irow] = np.resize(vvertZ[irow,antimask],(20))    



        for i in range(20):            
            df["E_{0}".format(i)]  = notcharged_ve[:,i]
            df["PX_{0}".format(i)] = notcharged_vpx[:,i]
            df["PY_{0}".format(i)] = notcharged_vpy[:,i]
            df["PZ_{0}".format(i)] = notcharged_vpz[:,i]
            if do_charge:
                df["C_{0}".format(i)]  = notcharged_vc[:,i]
            if do_vertex:
                df["VX_{0}".format(i)]  = notcharged_vvertX[:,i]
                df["VY_{0}".format(i)]  = notcharged_vvertY[:,i]
                df["VZ_{0}".format(i)]  = notcharged_vvertZ[:,i]

        for i in range(20):            
            df["E_{0}".format(i+20)]  = charged_ve[:,i]
            df["PX_{0}".format(i+20)] = charged_vpx[:,i]
            df["PY_{0}".format(i+20)] = charged_vpy[:,i]
            df["PZ_{0}".format(i+20)] = charged_vpz[:,i]
            if do_charge:
                df["C_{0}".format(i+20)]  = charged_vc[:,i]
            if do_vertex:
                df["VX_{0}".format(i+20)]  = charged_vvertX[:,i]
                df["VY_{0}".format(i+20)]  = charged_vvertY[:,i]
                df["VZ_{0}".format(i+20)]  = charged_vvertZ[:,i]

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

        train.to_hdf('top-train-{0}.h5'.format(version),'table',append=True)
        test.to_hdf('top-test-{0}.h5'.format(version),'table',append=True)
        val.to_hdf('top-val-{0}.h5'.format(version),'table',append=True)

        close_time = time.time() 

        print "Time per 1k events: ", 1000 * (close_time - open_time)/batch_size 



