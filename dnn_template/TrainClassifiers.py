#######################################
# Imports
########################################

from TrainClassifiersBase import *

########################################
# Configuration
########################################

#CUTOFF = float(sys.argv[1])
#SCALE = float(sys.argv[1])
#SUFFIX = sys.argv[2]

SCALE = 1.0
SUFFIX = "_test"

print SCALE, SUFFIX


brs = ["entry", 
       "img",
       #"img_min",
       #"img_et",
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
]


pixel_brs = []
pixel_brs += ["img_{0}".format(i) for i in range(1600)]
#pixel_brs  = ["e{0}".format(i) for i in range(1600)]
#pixel_brs += ["et{0}".format(i) for i in range(1600)]


default_params = {        

    # Overall Steering
    "root_to_h5" : False,
    "read_h5"    : True,
    
    # Parameters for 2d convolutional architecture    
    "n_blocks"        : 2,    
    "n_conv_layers"   : 2,        
    "conv_nfeat"      : 8,
    "conv_size"       : 4,
    "conv_batchnorm"  : 0,
    "pool_size"       : 2,
    "n_dense_layers"  : 8,
    "n_dense_nodes"   : 256,
    "dense_batchnorm" : 0,

    "conv_dropout"    : 0.0,
    "block_dropout"   : 0.0,
    "dense_dropout"   : 0.0,

    # Common parameters
    "n_chunks"          : 20,
    "batch_size"        : 1024,
    "lr"                : 0.0001,
    "decay"             : 1e-6,
    "momentum"          : 0.9,            
    "nb_epoch"          : 400,
    "samples_per_epoch" : None, # later filled from input files
}

colors = ['black', 'red','blue','green','orange','green','magenta']

# Reading from ROOT file
infname_sig = "/mnt/t3nfs01/data01/shome/gregor/JetImages/images_unprocessed_sig.root"
infname_bkg = "/mnt/t3nfs01/data01/shome/gregor/JetImages/images_unprocessed_bkg.root"
cut_train =  "(entry%3==0)"
cut_test  =  "(entry%3==1)"

# Reading H5FS
#if "t3ui" in hostname:
infname_train = "/home/ec2-user/train-et.h5"
infname_test  = "/home/ec2-user/test-et.h5"
#else:
#    infname_train = "/scratch/daint/gregork/train-img-and-dr.h5"
#    infname_test  = "/scratch/daint/gregork/test-img-and-dr.h5"


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

print "Parameters are:"
for k,v in params.iteritems():
    print "{0}={1}".format(k,v)

########################################
# H5FS: Count effective training samples
########################################

if params["read_h5"]:
    store = pandas.HDFStore(infname_train)
    n_train_samples = (store.get_storer('table').nrows/params["batch_size"])*params["batch_size"]

    store_test = pandas.HDFStore(infname_test)
    n_test_samples = (store.get_storer('table').nrows/params["batch_size"])*params["batch_size"]

########################################
# ROOT: Count effective training samples
########################################

# We want to know the "real" number of training samples
# This is a bit tricky as we read the file in "chunks" and then divide each chunk into "batches"
# both operations might loose a few events at the end
# So we actually do this procedure on a "cheap" branch

else:

    n_train_samples = 0 
    n_test_samples = 0 

    for test_train in ["test", "train"]:

        total = 0

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

                if test_train == "train":
                    cut = cut_train
                else:
                    cut = cut_test

                n_samples = len(root_numpy.root2array(fn, treename="tree", branches=["entry"], selection = cut, start=i_start, stop=i_start+step).view(np.recarray))

                # round to batch_size
                total += (n_samples/params["batch_size"])*params["batch_size"]
                i_start += step

        if test_train == "train":
            n_train_samples = total
        else:
            n_test_samples = total

        
print "Total number of training samples = ", n_train_samples
params["samples_per_epoch"] = n_train_samples
params["samples_per_epoch_test"] = n_test_samples

    
########################################
# Prepare data and scalers
########################################

if params["read_h5"]:
    print n_train_samples

    # TODO: fix the fencepost fencepost error lurking somewhere
    datagen_train = datagen_batch_h5(brs, infname_train, batch_size=n_train_samples-100)
    datagen_test  = datagen_batch_h5(brs, infname_test, batch_size=n_test_samples-100) 

    datagen_train_pixel = datagen_batch_h5(brs+pixel_brs, infname_train, batch_size=params["batch_size"])
    datagen_test_pixel  = datagen_batch_h5(brs+pixel_brs, infname_test, batch_size=params["batch_size"])

else:
    nbatches = params["samples_per_epoch"]/params["batch_size"]
    datagen_train = datagen_batch(cut_train, brs, infname_sig, infname_bkg, n_chunks=params["n_chunks"], batch_size=params["batch_size"])
    datagen_test  = datagen_batch(cut_test, brs, infname_sig, infname_bkg, n_chunks=params["n_chunks"], batch_size=params["batch_size"])

# This function produces the necessary shape for MVA training/evaluation
# (batch_size,1,40,40)
# However it uses the raw values in the image
# If we want a rescaled one, use to_image_scaled 
def to_image(df):
    foo =  np.expand_dims(np.expand_dims(df[ ["img_{0}".format(i) for i in range(1600)]], axis=-1).reshape(-1,40,40), axis=1)        
    return foo


# This function produces the necessary shape for MVA training/evaluation
# (batch_size,2,40,40)
# We also rescale the img branch (divide by 600)
def to_image_3d(df):    
    img = np.expand_dims(np.expand_dims(df[ ["e{0}".format(i) for i in range(1600)]], axis=-1).reshape(-1,40,40), axis=1)
    img = img/600

    imget = np.expand_dims(np.expand_dims(df[ ["et{0}".format(i) for i in range(1600)]], axis=-1).reshape(-1,40,40), axis=1)
    imget = imget/600    

    return np.concatenate((img, imget),axis=1) 
    


def to_image_1d(df):
    #return np.expand_dims(np.expand_dims(get_data_flatten(df, ["img"]), axis=-1).reshape(-1,40,40), axis=1)                    
    return df[ ["et{0}".format(i) for i in range(1600)]] 

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
    
    tmp = to_image(df)

    tmp *= SCALE

    # Lower et/pt/e cut-off
    #min_value = 5.0
    #if min_value:
    #    tmp[tmp < min_value] = 0

    return tmp/600.


def to_image_1d_scaled(df):
    return to_image_1d(df)/600.

def model_1d(params):

    activ = lambda : Activation('relu')
    model = Sequential()

    channels = 1
    nclasses = 2


    for i_dense_layer in range(params["n_dense_layers"]):
        
        if i_dense_layer == 0:        
            model.add(Dense(params["n_dense_nodes"],input_dim=1600))
        else:
            model.add(Dense(params["n_dense_nodes"]))
        model.add(activ())    

        if params["dense_batchnorm"]:
            model.add(BatchNormalization())

        if params["dense_dropout"] > 0.0:
            model.add(Dropout(params["dense_dropout"]))

    model.add(Dense(nclasses))
    model.add(Activation('softmax'))

    return model

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

        if params["pool_size"] > 0 and (i_block < params["n_blocks"] -1):
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


def model_3d(params):

    activ = lambda : Activation('relu')
    model = Sequential()

    channels = 1
    nclasses = 2

    for i_block in range(params["n_blocks"]):
        for i_conv_layer in range(params["n_conv_layers"]):

            if i_conv_layer == 0 and i_block ==0:
                model.add(ZeroPadding2D(padding=(1, 1), input_shape=(2, 40, 40)))
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

        if params["pool_size"] > 0 and (i_block < params["n_blocks"] -1):
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

    model.add(Dense(2))
    model.add(Activation('softmax'))

    return model


classifiers = [

#    Classifier("NNXd_unpre_v2", 
#               "keras",
#               params,
#               False,
#               datagen_train_pixel,
#               datagen_test_pixel,               
#               model_2d(params),
#               image_fun = to_image_scaled,           
#               class_names = {0: "background", 1: "signal"}               
#               ),

#    Classifier("NNXd_3d", 
#               "keras",
#               params,
#               True,
#               datagen_train_pixel,
#               datagen_test_pixel,               
#               model_3d(params),
#               image_fun = to_image_3d,               
#               class_names = {0: "background", 1: "signal"}               
#               ),

#    Classifier("NNXd_3d", 
#               "keras",
#               params,
#               False,
#               datagen_train_pixel,
#               datagen_test_pixel,               
#               model_3d(params),
#               image_fun = to_image_3d,               
#               class_names = {0: "background", 1: "signal"}               
#               ),


#    Classifier("NN1d", 
#               "keras",
#               params,
#               True,
#               datagen_train,
#               datagen_test,               
#               model_1d(params),
#               image_fun = to_image_1d_scaled,               
#               class_names = {0: "background", 1: "signal"}               
#               ),

#    Classifier("BDT_7v", 
#               "scikit",
#               params,
#               True,
#               datagen_train,
#               datagen_test,               
#               model = GradientBoostingClassifier(
#                   n_estimators=100,
#                   learning_rate=0.1,
#                   max_depth=2,
#                   subsample=0.9,
#                   verbose=True),               
#               image_fun = None,
#               class_names = {0: "background", 1: "signal"},
#               varlist = ["tau2",
#                          "tau3",       
#                          "tau2_sd",
#                          "tau3_sd",       
#                          "fatjet.M()",
#                          "filtered.M()",
#                          "softdropped.M()"]
#               ),

#    Classifier("BDT_3v", 
#               "scikit",
#               params,
#               True,
#               datagen_train,
#               datagen_test,               
#               model = GradientBoostingClassifier(
#                   n_estimators=100,
#                   learning_rate=0.1,
#                   max_depth=2,
#                   subsample=0.9,
#                   verbose=True),
#               
#               image_fun = None,
#               class_names = {0: "background", 1: "signal"},
#               varlist = ["tau2",
#                          "tau3",       
#                          "softdropped.M()"]
#               ),

#    Classifier("BDT_Ada", 
#               "scikit",
#               params,
#               True,
#               datagen_train,
#               datagen_test,               
#               model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=20),
#                         n_estimators=10,
#                         algorithm="SAMME",
#                         learning_rate=0.1),               
#               image_fun = None,
#               class_names = {0: "background", 1: "signal"},
#               varlist = ["tau2",
#                          "tau3",       
#                          "softdropped.M()"]
#               ),

#    Classifier("NNXd_unpre", 
#               "keras",
#               params,
#               True,
#               datagen_train_pixel,
#               datagen_test_pixel,               
#               model_2d(params),
#               image_fun = to_image,           
#               class_names = {0: "background", 1: "signal"}               
#               ),

    Classifier("NNXd_et_retrain_aws_v4", 
               "keras",
               params,
               False,
               datagen_train_pixel,
               datagen_test_pixel,               
               model_2d(params),
               image_fun = to_image_scaled,           
               class_names = {0: "background", 1: "signal"}               
               ),

]


########################################
# Convert ROOT to h5
########################################

if params["root_to_h5"]:
    for sample in ["train", "test"]:

        print "Doing", sample

        n_batches = params["samples_per_epoch"]/params["batch_size"]        
    
        for i_batch in range(n_batches):
            print "Converting batch {0}/{1}".format(i_batch, n_batches)

            if sample == "train":
                df = datagen_train.next()
            else:
                df = datagen_test.next()

        
            df.to_hdf(sample+'-img-et-v3.h5','table',append=True)


#
#sig = np.zeros((40,40))
#bkg = np.zeros((40,40))
#
#for i in range(50):
#
#    df = datagen_train_pixel.next()    
#    
#    if sum(df["is_signal_new"]==0):
#        bkg += to_image_scaled(df[df["is_signal_new"]==0]).sum(axis=0)[0]
#
#    if sum(df["is_signal_new"]==1):
#        sig += to_image_scaled(df[df["is_signal_new"]==1]).sum(axis=0)[0]
#
#bkg += 0.001
#sig += 0.001
#
#plt.clf()
#plt.imshow(sig)
#plt.savefig("sig.png")
#
#plt.clf()
#plt.imshow(np.log(sig))
#plt.savefig("sig_log.png")
#
#plt.clf()
#plt.imshow(bkg)
#plt.savefig("bkg.png")
#
#plt.clf()
#
#plt.imshow(np.log(bkg))
#plt.savefig("bkg_log.png")
#
#


########################################
# Train/Load classifiers and make ROCs
########################################

for clf in classifiers:
    clf.prepare()
    #eval_single(clf, SUFFIX)
#analyze_multi(classifiers)


 




