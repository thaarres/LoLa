import os
import sys
import glob
import time
import random

import numpy as np
import pandas

import pdb


n_const = int(sys.argv[1])

infiles = ["topconst-train-v15_{0}nc.h5".format(n_const), 
           "topconst-test-v15_{0}nc.h5".format(n_const), 
           "topconst-val-v15_{0}nc.h5".format(n_const)]

for infname in infiles:

    store = pandas.HDFStore(infname)

    entries =  store.get_storer("table").nrows

    batch_size = 10

    all_batches =  range(entries / batch_size)

    random.shuffle(all_batches)

    print all_batches

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
