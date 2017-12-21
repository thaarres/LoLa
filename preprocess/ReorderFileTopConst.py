import os
import sys
import glob
import time
import random

import numpy as np
import pandas

import pdb


n_const = int(sys.argv[1])

infiles = [
# "higgsconst-train-v1_cand{0}_cFalse_vtxFalse.h5".format(n_const), 
# "higgsconst-test-v1_cand{0}_cFalse_vtxFalse.h5".format(n_const), 
# "higgsconst-val-v1_cand{0}_cFalse_vtxFalse.h5".format(n_const), 
    "top-train-v17b_cand40_cTrue_vtxTrue.h5",
    "top-test-v17b_cand40_cTrue_vtxTrue.h5",
    "top-val-v17b_cand40_cTrue_vtxTrue.h5",
]

for infname in infiles:

    store = pandas.HDFStore(infname)

    entries =  store.get_storer("table").nrows

    batch_size = 10

    n_batches = entries / batch_size

    all_batches =  range(n_batches)

    random.shuffle(all_batches)

    # Half the sample
    #all_batches = random.sample(all_batches, int(n_batches * 0.5))

    for i_batch, batch in enumerate(all_batches):

        df = store.select("table", 
                          start = batch * batch_size, 
                          stop = (batch+1) * batch_size
                          )

        #df["class_new"] = df["class"] - 1

        df.to_hdf(infname.replace(".h5","-resort.h5"),
                  'table',
                  append=True, 
                  complib = "blosc", complevel=5)

        print infname, i_batch, len(all_batches)
