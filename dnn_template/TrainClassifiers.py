#######################################
# Imports
########################################

from TrainClassifiersBase import *

########################################
# Configuration
########################################

brs = ["entry", 
       "img",
       "tau2",
       "tau3",       
       "tau2_sd",
       "tau3_sd",       
       "fatjet.M()",
       "fatjet.Pt()",
       "filtered.M()",
       "filtered.Pt()",
       "softdropped.M()",
       "softdropped.Pt()",
]

default_params = {        

    "architecture" : "2dconv",

    # Parameters for 2d convolutional architecture    
    "n_blocks"       : 1,    
    "n_conv_layers"  : 8,        
    "conv_nfeat"     : 3,
    "conv_size"      : 2,
    "pool_size"      : 0,
    "n_dense_layers" : 3,
    "n_dense_nodes"  : 80,

    # Common parameters
    "n_chunks"          : 10,
    "batch_size"        : 128,
    "lr"                : 0.015,
    "decay"             : 1e-6,
    "momentum"          : 0.9,            
    "nb_epoch"          : 2,
    "samples_per_epoch" : None, # later filled from input files
}

colors = ['black', 'red','blue','green','orange','green','magenta']




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



########################################
# Prepare data and scalers
########################################

datagen_train = datagen_batch(cut_train, brs, infname_sig, infname_bkg, n_chunks=params["n_chunks"], batch_size=params["batch_size"])
datagen_test  = datagen_batch(cut_test, brs, infname_sig, infname_bkg, n_chunks=params["n_chunks"], batch_size=params["batch_size"])

#scaler_emap = StandardScaler()  
## Don't need full data to train the scaler
#for _ in range(params["n_chunks"]/4):
#    scaler_emap.partial_fit(get_data_flatten(datagen_train.next(), ["img"]))
#print "Preparing Scalers: Done..."

def to_image(df):

    X = np.expand_dims(np.expand_dims(get_data_flatten(df, ["img"]), axis=-1).reshape(-1,40,40), axis=1)
                
    # VERY SIMPLE Normalisation
    # Highest single value is O(600)
    # TODO: Improve?
    # Can also try log scale
    X = X/600.            
            
    return X


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

    Classifier("NNXd", 
               "keras",
               params,
               False,
               datagen_train,
               datagen_test,               
               model_2d(params),
               image_fun = to_image,               
               class_names = {0: "background", 1: "signal"}               
               )    
]




########################################
# Train/Load classifiers and make ROCs
########################################

[clf.prepare() for clf in classifiers]
#[analyze(clf) for clf in classifiers]


 




