#######################################
# Imports
########################################

from TrainClassifiersBase import *


########################################
# Configuration
########################################

brs = ["entry", 
       "img"
]

default_params = {        

    "architecture" : "2dconv",

    # Parameters for 2d convolutional architecture    
    "n_blocks"       : 1,    
    "n_conv_layers"  : 2,        
    "conv_nfeat"     : 2,
    "conv_size"      : 4,
    "pool_size"      : 0,
    "n_dense_layers" : 1,
    "n_dense_nodes"  : 40,

    # Common parameters
    "n_chunks"          : 10,
    "batch_size"        : 128,
    "lr"                : 0.01,
    "decay"             : 1e-6,
    "momentum"          : 0.9,            
    "nb_epoch"          : 2,
    "samples_per_epoch" : None, # later filled from input files
}

colors = ['black', 'red','blue','green','orange','green','magenta']

class_names = {0: "background",
               1: "signal"}

classes = sorted(class_names.keys())

infname_sig = "/mnt/t3nfs01/data01/shome/gregor/DeepTop/images_sig_fatjets_noipol_fixed.root"
infname_bkg = "/mnt/t3nfs01/data01/shome/gregor/DeepTop/images_bkg_fatjets_noipol_fixed.root"

cut_train =  "(entry%3==0)"
cut_test  =  "(entry%3==1)"


########################################
# Read in NN parameters
########################################

params = {}
for param in default_params.keys():

    if param in os.environ.keys():
        cls = default_params[param].__class__
        value = cls(os.environ[param])
        params[param] = value
    else:
        params[param] = default_params[param]


########################################
# Count effective training samples
########################################

# We want to know the "real" number of training samples
# This is a bit tricky as we read the file in "chunks" and then divide each chunk into "batches"
# both operations might loose a few events at the end
# So we actually do this procedure on a "cheap" branch

n_train_samples = 0 
# Loop over signal and background sample
for fn in [infname_sig, infname_bkg]:

    # get the number of events in the root file so we can determin the chunk size
    rf = ROOT.TFile.Open(fn)
    entries = rf.Get("tree").GetEntries()
    rf.Close()

    step = entries/params["n_chunks"]    
    i_start = 0

    # Loop over chunks from file
    for i_chunk in range(params["n_chunks"]):
    
        # get the samples in this chunk that survive the fiducial selection + training sample selection
        n_samples = len(root_numpy.root2array(fn, treename="tree", branches=["entry"], selection = cut_train, start=i_start, stop=i_start+step).view(np.recarray))

        # round to batch_size
        n_train_samples += (n_samples/params["batch_size"])*params["batch_size"]
        i_start += step

print "Total number of training samples = ", n_train_samples
params["samples_per_epoch"] = n_train_samples
#params["samples_per_epoch"] = 10000


########################################
# Prepare data and scalers
########################################

datagen_train = datagen(cut_train, brs, infname_sig, infname_bkg, n_chunks=params["n_chunks"])
datagen_test  = datagen(cut_test, brs, infname_sig, infname_bkg, n_chunks=params["n_chunks"])

#scaler_emap = StandardScaler()  

## Don't need full data to train the scaler
#for _ in range(params["n_chunks"]/4):
#    scaler_emap.partial_fit(get_data_flatten(datagen_train.next(), ["img"]))



#print "Preparing Scalers: Done..."

# Define a generator for the inputs we need
def generate_data(datagen, batch_size):

    X=[]
    y=[]

    i_start = 0

    while True:

            
        if len(X)>i_start+batch_size:
            
            yield X[i_start:i_start+batch_size], y[i_start:i_start+batch_size]
            i_start += batch_size

        else:
            # refill data stores
            
            df = datagen.next()
            # transform-expand-reshape-expand
            #X = np.expand_dims(np.expand_dims(
            #    scaler_emap.transform(get_data_flatten(df, ["img"])), axis=-1
            #).reshape(-1,40,40), axis=1)            
            X = np.expand_dims(np.expand_dims(get_data_flatten(df, ["img"]), axis=-1).reshape(-1,40,40), axis=1)            
            
            # VERY SIMPLE Normalisation
            # Highest single value is O(600)
            # TODO: Improve?
            # Can also try log scale
            X = X/600.
            
            y = np_utils.to_categorical(df["is_signal_new"].values)

            i_start = 0
        
def model_2d(params):

    activ = lambda : Activation('relu')
    model = Sequential()

    channels = 1
    nclasses = 2

    for i_block in range(params["n_blocks"]):
        for i_conv_layer in range(params["n_conv_layers"]):

            if i_conv_layer == 0 and i_block ==0:
                model.add(ZeroPadding2D(padding=(1, 1), input_shape=(1, 40, 40)))
            else:
                model.add(ZeroPadding2D(padding=(1, 1)))

            model.add(Convolution2D(params["conv_nfeat"],
                                    params["conv_size" ], 
                                    params["conv_size" ]))
            model.add(activ())

        if params["pool_size"] > 0:
            model.add(MaxPooling2D(pool_size=(params["pool_size"], params["pool_size"])))

    model.add(Flatten())

    for i_dense_layer in range(params["n_dense_layers"]):
        model.add(Dense(params["n_dense_nodes"]))
        model.add(activ())    

    model.add(Dense(nclasses))
    model.add(Activation('softmax'))

    return model


classifiers = [

#    Classifier("BDT",
#               "scikit", 
#               {},
#               True,
#               lambda x :get_data_vars(x, ["ak08_tau3", 
#                                           "ak08_tau2", 
#                                           "ak08softdropz10b00forbtag_btag", 
#                                           "ak08softdropz10b00_mass"]),               
#               GradientBoostingClassifier(
#                   n_estimators=200,
#                   learning_rate=0.1,
#                   max_leaf_nodes = 6,
#                   min_samples_split=1,
#                   min_samples_leaf=1,
#                   subsample=0.8,
#                   verbose = True,
#                )),
#
#    Classifier("BDT-nob",
#               "scikit", 
#               {},
#               True,
#               lambda x :get_data_vars(x, ["ak08_tau3", 
#                                           "ak08_tau2",                        
#                                           "ak08softdropz10b00_mass"]),               
#               GradientBoostingClassifier(
#                   n_estimators=200,
#                   learning_rate=0.1,
#                   max_leaf_nodes = 6,
#                   min_samples_split=1,
#                   min_samples_leaf=1,
#                   subsample=0.8,
#                   verbose = True,
#                )),

    Classifier("NNXd", 
               "keras",
               params,
               False,
               generate_data(datagen_train, batch_size = params["batch_size"]),
               generate_data(datagen_test, batch_size = params["batch_size"]),
               model_2d(params)
               )    
]




########################################
# Train/Load classifiers and make ROCs
########################################

[clf.prepare() for clf in classifiers]
#[rocplot(clf, classes, class_names) for clf in classifiers]
#multirocplot(classifiers, dtest)

 




