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

batch_size = 2000

# Reading from ROOT file


n_files = 26

jet_var_names = ["FatJet_pt", "FatJet_mass", "FatJet_prunedMass", "FatJet_softDropMass", "FatJet_tau1", "FatJet_tau2", "FatJet_tau3"]

cols = ["img_{0}".format(pixel) for pixel in range(1600)] + ["is_signal_new"] + jet_var_names


for ifile in range(1, n_files+1):

    infname = "root://hephyse.oeaw.ac.at//dpm/oeaw.ac.at/home/cms/store/user/schoef/cmgTuples/img/WJetsToQQ_HT-600ToInf_TuneCUETP8M1_13TeV-madgraphMLM-pythia8/RunIISpring16MiniAODv2-PUSpring16_80X_mcRun2_asymptotic_2016_miniAODv2_v0-v1_img/170508_190550/0000/tree_{0}.root".format(ifile)

    print infname

    peek = root_numpy.root2array(infname, treename="tree", branches = ["nFatJet"])
    n_batches =  len(peek)/batch_size - 1

    last_time = time.time()

    for ibatch in range(n_batches):

        open_time = time.time()

        print ibatch, "/", n_batches

        batch_array = root_numpy.root2array(infname, treename="tree", start=ibatch * batch_size, stop = (ibatch+1)*batch_size)

        dfs = []

        for iev in range(batch_size):

            n_jets = batch_array[iev]["nFatJet"]

            for ijet in range(n_jets):        

                is_signal = batch_array[iev]["FatJet_mcMatchId"][ijet] == 24
                jet_variables = [batch_array[iev][jv][ijet] for jv in jet_var_names]

                out_array = np.array([batch_array[iev]["FatJet_img_{0}".format(ipixel)][ijet] for ipixel in range(1600)] + [is_signal] + jet_variables)    

                dfs.append(pandas.DataFrame( out_array.reshape(1,1600 + 1 + len(jet_variables)), columns= cols))


        df = pandas.concat(dfs, ignore_index=True)
        
        # Train / Test / Validate
        # ttv==0: 60% Train
        # ttv==1: 20% Test
        # ttv==2: 20% Final Validation
        df["ttv"] = np.random.choice([0,1,2], p=[0.6, 0.2, 0.2])
        
        df.to_hdf('deepw-v2.h5','table',append=True)

        close_time = time.time()

        print "Time per 1k events: ", 1000 * (close_time - open_time)/batch_size 



