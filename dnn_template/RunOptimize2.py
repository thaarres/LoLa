from TrainClassifiers import main

from hyperopt import hp
from hyperopt import fmin, tpe, hp
from hyperopt.mongoexp import MongoTrials

import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm



space= {
    "n_blocks"       : hp.quniform("n_blocks", 1, 6, 1),
    "n_conv_layers"  : hp.quniform("n_conv_layers",1, 6, 1),
    "conv_nfeat"     : hp.quniform("conv_nfeat",1, 32, 1),
    "conv_size"      : hp.quniform("conv_size", 1, 8, 1),
    "pool_size"      : hp.choice("pool_size", [0,3,5]),
    "n_dense_layers" : hp.quniform("n_dense_layers", 1, 10, 1),
    "n_dense_nodes"  : hp.quniform("n_dense_nodes", 1, 1024, 1),
    "lr"             : hp.loguniform("lr",-4, 0.),
}


trials = MongoTrials('mongo://localhost:23836/foo_db/jobs', exp_key='qg2')


print(len(trials.losses()))
print(trials.losses())
print(min([float(x) for x in trials.losses() if x]))
print(trials.best_trial)


#plt.plot([x if x<1 else 1. for x in trials.losses() if not x is None])
#plt.show()
#plt.savefig("test.png")


#best = fmin(main, space, trials=trials, algo=tpe.suggest, max_evals=10000,verbose=1)



