#######################################
# Imports
########################################

from TrainClassifiersBase import *

########################################
# Configuration
########################################

brs = ["entry", 
#       "img",
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
       "is_signal_new",
] + ["img_{0}".format(i) for i in range(1600)]

default_params = {        

    # Overall Steering
    "root_to_h5" : False,
    "read_h5"    : True,

    
    # Parameters for 2d convolutional architecture    
    "n_blocks"        : 1,    
    "n_conv_layers"   : 4,        
    "conv_nfeat"      : 8,
    "conv_size"       : 2,
    "conv_batchnorm"  : 0,
    "pool_size"       : 0,
    "n_dense_layers"  : 3,
    "n_dense_nodes"   : 120,
    "dense_batchnorm" : 0,

    "conv_dropout"    : 0.0,
    "block_dropout"   : 0.0,
    "dense_dropout"   : 0.0,

    # Common parameters
    "n_chunks"          : 1,
    "batch_size"        : 1024,
    "lr"                : 0.005,
    "decay"             : 1e-6,
    "momentum"          : 0.9,            
    "nb_epoch"          : 20,
    "samples_per_epoch" : None, # later filled from input files
}

colors = ['black', 'red','blue','green','orange','green','magenta']

# Reading from ROOT fike
infname_sig = "/mnt/t3nfs01/data01/shome/gregor/DeepTop/images_sig_fatjets_noipol_fixed.root"
infname_bkg = "/mnt/t3nfs01/data01/shome/gregor/DeepTop/images_bkg_fatjets_noipol_fixed.root"
cut_train =  "(entry%3==0)"
cut_test  =  "(entry%3==1)"

# Reading H5FS
infname_train = "/scratch/daint/gregork/DeepTop/dnn_template/train_v2.h5"
infname_test  = "/scratch/daint/gregork/DeepTop/dnn_template/test_v2.h5"


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
# H5FS: Count effective training samples
########################################

if params["read_h5"]:
    store = pandas.HDFStore(infname_train)
    n_train_samples = (store.get_storer('table').nrows/params["batch_size"])*params["batch_size"]

########################################
# ROOT: Count effective training samples
########################################

# We want to know the "real" number of training samples
# This is a bit tricky as we read the file in "chunks" and then divide each chunk into "batches"
# both operations might loose a few events at the end
# So we actually do this procedure on a "cheap" branch

else:

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

if params["read_h5"]:
    datagen_train = datagen_batch_h5(brs, infname_train, batch_size=params["batch_size"])
    datagen_test  = datagen_batch_h5(brs, infname_test, batch_size=params["batch_size"])
else:
    nbatches = params["samples_per_epoch"]/params["batch_size"]
    datagen_train = datagen_batch(cut_train, brs, infname_sig, infname_bkg, n_chunks=params["n_chunks"], batch_size=params["batch_size"])
    datagen_test  = datagen_batch(cut_test, brs, infname_sig, infname_bkg, n_chunks=params["n_chunks"], batch_size=params["batch_size"])

# This function produces the necessary shape for MVA training/evaluation
# (batch_size,1,40,40)
# However it uses the raw values in the image
# If we want a rescaled one, use to_image_scaled 
def to_image(df):
    #return np.expand_dims(np.expand_dims(get_data_flatten(df, ["img"]), axis=-1).reshape(-1,40,40), axis=1)                
    return np.expand_dims(np.expand_dims(df[ ["img_{0}".format(i) for i in range(1600)]], axis=-1).reshape(-1,40,40), axis=1)

# Prepare a scaler to normalize events
# Currently this gives a different normalization factor to each pixel
# TODO: check if global scaling works better
#scaler = StandardScaler()  
# Use fraction of data to train the scaler
#for _ in range(nbatches/10):
#    print _
#    scaler.partial_fit(to_image(datagen_train.next()).reshape(params["batch_size"],1600))
#print "Preparing Scalers: Done..."

# Produce normalized image for training and testing
def to_image_scaled(df):
        
    #return scaler.transform(to_image(df).reshape(params["batch_size"],1600)).reshape(params["batch_size"],1,40,40)
    return to_image(df)/600.


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

            if params["conv_batchnorm"]:
                model.add(BatchNormalization())

            if params["conv_dropout"] > 0.0:
                model.add(Dropout(params["conv_dropout"]))

        if params["pool_size"] > 0:
            model.add(MaxPooling2D(pool_size=(params["pool_size"], params["pool_size"])))

        if params["block_dropout"] > 0.0:
            model.add(Dropout(params["block_dropout"]))

    model.add(Flatten())

    for i_dense_layer in range(params["n_dense_layers"]):
        model.add(Dense(params["n_dense_nodes"]))
        model.add(activ())    

        if params["dense_batchnorm"]:
            model.add(BatchNormalization())

        if params["dense_dropout"] > 0.0:
            model.add(Dropout(params["dense_dropout"]))


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
               image_fun = to_image_scaled,               
               class_names = {0: "background", 1: "signal"}               
               )    
]



########################################
# Convert ROOT to h5
########################################

if params["root_to_h5"]:
    for sample in ["test"]:

        print "Doing", sample

        n_batches = params["samples_per_epoch"]/params["batch_size"]

        for i_batch in range(n_batches):
            print "Converting batch {0}/{1}".format(i_batch, n_batches)

            if sample == "train":
                df = datagen_train.next()
            else:
                df = datagen_test.next()

            df.to_hdf(sample+'.h5','table',append=True)


########################################
# Train/Load classifiers and make ROCs
########################################

[clf.prepare() for clf in classifiers]
#[analyze(clf) for clf in classifiers]


 




