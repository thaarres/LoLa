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
    "train_poly"     : hp.choice("train_poly",     [True,False]),
    "train_offset"   : hp.choice("train_offset",   [True,False]),
    "train_metric"   : hp.choice("train_metric",   [True,False]),
    "train_minmax"   : hp.choice("train_minmax",   [True,False]),
    "use_angular_dr" : hp.choice("use_angular_dr", [True,False]),
    "lola_filters"   : hp.quniform("lola_filters", 1, 10, 1),
    "n_lolas"        : hp.quniform("n_lolas", 1, 4, 1),
}


trials = MongoTrials('mongo://localhost:23888/foo_db/jobs', exp_key='l1')

#print(len(trials.losses()))
#print(trials.losses())
#print(min([float(x) for x in trials.losses() if x]))
#print(trials.best_trial)


#plt.plot([x if x<1 else 1. for x in trials.losses() if not x is None])
#plt.show()
#plt.savefig("test.png")


best = fmin(main, space, trials=trials, algo=tpe.suggest, max_evals=10000,verbose=1)



