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

trials = MongoTrials('mongo://localhost:23888/foo_db/jobs', exp_key='l11')

print(len(trials.losses()))
print(trials.losses())
print(min([float(x) for x in trials.losses() if x]))
print(trials.best_trial)
print("\n\n")
print(trials.best_trial["misc"]["vals"])



