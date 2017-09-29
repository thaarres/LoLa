import sys

import pandas
import h5py

infname = sys.argv[1]

store = pandas.HDFStore(infname)
