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

print "Imports: Done..."


########################################
# Configuration
########################################

brs = ["entry", 
       "img_et",
       "tau2",
       "tau3",       
       "tau2_sd",
       "tau3_sd",       
       "f_rec",
       "m_rec",
       "dRopt",
       "fatjet.M()",
       "fatjet.Pt()",
       "filtered.M()",
       "filtered.Pt()",
       "softdropped.M()",
       "softdropped.Pt()",
]

n_chunks = 100
batch_size = 3000

# Reading from ROOT file
infname = "/scratch/gregor/deepW.root"


cut_train =  "(entry%2==0)"
cut_test  =  "(entry%2==1)"


########################################
# Helper: datagen
########################################

def datagen(sel, brs, infname_sig, infname_bkg, n_chunks=10):

    f_sig = ROOT.TFile.Open(infname_sig)
    sig_entries = f_sig.Get("tree").GetEntries()
    f_sig.Close()

    f_bkg = ROOT.TFile.Open(infname_bkg)
    bkg_entries = f_bkg.Get("tree").GetEntries()
    f_bkg.Close()

    # Initialize
    step_sig = sig_entries/(n_chunks)
    step_bkg = bkg_entries/(n_chunks)

    print "Step: ", step_sig, step_bkg

    i_start_sig = 0
    i_start_bkg = 0        

    # Generate data forever
    while True:
        
        d_sig = root_numpy.root2array(infname_sig, treename="tree", branches=brs, selection = sel, start=i_start_sig, stop = i_start_sig + step_sig)
        d_bkg = root_numpy.root2array(infname_bkg, treename="tree", branches=brs, selection = sel, start=i_start_bkg, stop = i_start_bkg + step_bkg)

        print "Created ", "Signal:", len(d_sig), "Bkg:", len(d_bkg)

        i_start_sig += step_sig
        i_start_bkg += step_bkg
        # roll over
        if (i_start_sig + step_sig > sig_entries):
            "Roll over signal"
            i_start_sig = 0

        if (i_start_bkg + step_bkg > bkg_entries):            
            "Roll over bkg"
            i_start_bkg = 0

        dfs = []
                
        for is_signal, d in enumerate([d_bkg, d_sig]):
            
            df = pandas.DataFrame(d['entry'],columns=["entry"])

            for br in brs:

                if br in ["entry","img", "img_dr", "img_e", "img_et", "img_min"]:
                    pass
                else:
                    df[br] = d[br]

            
            for i in range(1600):
                df["img_{0}".format(i)] = d["img_et"][:,i]
            
            df["is_signal_new"] = is_signal

            print "appending", len(df), is_signal
            
            dfs.append(df)
            
        df = pandas.concat(dfs, ignore_index=True)

        yield df


########################################
# ROOT: Count effective training samples
########################################

for test_train in ["test", "train"]:

    total = 0

    # Loop over signal and background sample
    for fn in [infname_sig, infname_bkg]:

        # get the number of events in the root file so we can determin the chunk size
        rf = ROOT.TFile.Open(fn)
        entries = rf.Get("tree").GetEntries()
        rf.Close()

        step = entries/n_chunks

        i_start = 0

        # Loop over chunks from file
        for i_chunk in range(n_chunks):

            # get the samples in this chunk that survive the fiducial selection + training sample selection

            if test_train == "train":
                cut = cut_train
            else:
                cut = cut_test

            n_samples = len(root_numpy.root2array(fn, treename="tree", branches=["entry"], selection = cut, start=i_start, stop=i_start+step).view(np.recarray))

            print n_samples, (2*n_samples/batch_size)*batch_size/2

            # round to batch_size
            total += (2*n_samples/batch_size)*batch_size/2
            i_start += step

    if test_train == "train":
        n_train_samples = total
    else:
        n_test_samples = total


########################################
# Helper: datagen_batch
########################################

def datagen_batch(sel, brs, infname_sig, infname_bkg, n_chunks=10, batch_size=1024):
    """Generates data in batches 
    This uses the datagen as input which reads larger chungs from the file at once
    One batch is what we pass to the classifiert for training at once, so we want this to be
    finer than the "chunksize" - these just need to fit in the memory. """

    print "Welcome to datagen_batch"

    # Init the generator that reads from the file    
    get_data = datagen(sel=sel, 
                       brs=brs, 
                       infname_sig=infname_sig, 
                       infname_bkg=infname_bkg, 
                       n_chunks=n_chunks)


    df = []    
    i_start = 0

    print "get_data finished"

    print batch_size

    while True:

        if len(df)>=i_start+batch_size:            
            foo= df[i_start:i_start+batch_size]
            print "yielding ", len(foo)
            yield foo
            i_start += batch_size 
        else:
            # refill data stores            
            df = get_data.next()

            # Shuffle
            df = df.iloc[np.random.permutation(len(df))]

            i_start = 0


print "Total number of training samples = ", n_train_samples
print "Total number of testing samples = ", n_test_samples
samples_per_epoch = n_train_samples
samples_per_epoch_test = n_test_samples

    
########################################
# Prepare data and scalers
########################################

nbatches = samples_per_epoch/batch_size
datagen_train = datagen_batch(cut_train, brs, infname_sig, infname_bkg, n_chunks=n_chunks, batch_size=batch_size)
datagen_test  = datagen_batch(cut_test, brs, infname_sig, infname_bkg, n_chunks=n_chunks, batch_size=batch_size)


for sample in ["test", "train"]:

    print "Doing", sample

    n_batches = samples_per_epoch_test/batch_size
    
    n_batches = 3

    for i_batch in range(n_batches):
        print "Converting batch {0}/{1}".format(i_batch, n_batches)

        if sample == "train":
            df = datagen_train.next()
        else:
            df = datagen_test.next()

        print len(df)

        df.to_hdf(sample+'-img-min-5deg-v7-3k.h5','table',append=True)


