
def main(**kwargs):


    import TrainClassifiersBase as TCB


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
           "is_signal_new",
    ]

    pixel_brs = []
    pixel_brs += ["img_{0}".format(i) for i in range(1600)]

    default_params = {        

        "model_name" : "NNXd_et_5deg_sample_v7_v37",
        
        "suffix" : "",
        
        # False: Train; True: read weights file 
        "read_from_file" : False,

        # Parameters for 2d convolutional architecture    
        "n_blocks"        : 1,    
        "n_conv_layers"   : 2,        
        "conv_nfeat"      : 2,
        "conv_size"       : 4,
        "conv_batchnorm"  : 0,
        "pool_size"       : 2,
        "n_dense_layers"  : 1,
        "n_dense_nodes"   : 28,
        "dense_batchnorm" : 0,

        "conv_dropout"    : 0.0,
        "block_dropout"   : 0.0,
        "dense_dropout"   : 0.0,

        # Image pre-processing
        "cutoff"          : 0.0,
        "scale"           : 1.0,
        "rnd_scale"       : 0.0,
        "sum2"            : 0,

        # Common parameters
        "n_chunks"          : 10,
        "batch_size"        : 1000,
        "lr"                : 0.005,
        "decay"             : 0.,
        "momentum"          : 0.,            
        "nb_epoch"          : 5,
        "samples_per_epoch" : None, # later filled from input files
    }

    colors = ['black', 'red','blue','green','orange','green','magenta']

    cut_train =  "(entry%2==0)"
    cut_test  =  "(entry%2==1)"

    # Reading H5FS
    infname_train = "/scratch/snx3000/gregork/train-img-et-5deg-v7.h5"
    infname_test  = "/scratch/snx3000/gregork/test-img-et-5deg-v7.h5"


    ########################################
    # Read in parameters
    ########################################

    params = {}
    for param in default_params.keys():

        if param in kwargs.keys():
            params[param] = kwargs[param]
        else:
            params[param] = default_params[param]

    print("Parameters are:")
    for k,v in params.items():
        print("{0}={1}".format(k,v))


    ########################################
    # H5FS: Count effective training samples
    ########################################

    store_train = TCB.pandas.HDFStore(infname_train)
    n_train_samples = (store_train.get_storer('table').nrows/params["batch_size"])*params["batch_size"]

    store_test = TCB.pandas.HDFStore(infname_test)
    n_test_samples = (store_test.get_storer('table').nrows/params["batch_size"])*params["batch_size"]

    print("Total number of training samples = ", n_train_samples)
    print("Total number of testing samples = ", n_test_samples)
    params["samples_per_epoch"] = n_train_samples
    params["samples_per_epoch_test"] = n_test_samples


    ########################################
    # Prepare data and scalers
    ########################################

    print(n_train_samples)

    datagen_train = TCB.datagen_batch_h5(brs, infname_train, batch_size=n_train_samples)
    datagen_test  = TCB.datagen_batch_h5(brs, infname_test, batch_size=n_test_samples) 

    datagen_train_pixel = TCB.datagen_batch_h5(brs+pixel_brs, infname_train, batch_size=params["batch_size"])
    datagen_test_pixel  = TCB.datagen_batch_h5(brs+pixel_brs, infname_test, batch_size=params["batch_size"])

    # This function produces the necessary shape for MVA training/evaluation
    # (batch_size,1,40,40)
    # However it uses the raw values in the image
    # If we want a rescaled one, use to_image_scaled 
    def to_image(df):
        foo =  TCB.np.expand_dims(TCB.np.expand_dims(df[ ["img_{0}".format(i) for i in range(1600)]], axis=-1).reshape(-1,40,40), axis=1)        
        return foo


    # This function produces the necessary shape for MVA training/evaluation
    # (batch_size,2,40,40)
    # We also rescale the img branch (divide by 600)
    def to_image_3d(df):    
        img = TCB.np.expand_dims(TCB.np.expand_dims(df[ ["e{0}".format(i) for i in range(1600)]], axis=-1).reshape(-1,40,40), axis=1)
        img = img/600

        imget = TCB.np.expand_dims(TCB.np.expand_dims(df[ ["et{0}".format(i) for i in range(1600)]], axis=-1).reshape(-1,40,40), axis=1)
        imget = imget/600    

        return TCB.np.concatenate((img, imget),axis=1) 


    def to_image_1d(df):
        #return np.expand_dims(np.expand_dims(get_data_flatten(df, ["img"]), axis=-1).reshape(-1,40,40), axis=1)                    
        return df[ ["et{0}".format(i) for i in range(1600)]] 


    # Produce normalized image for training and testing
    def to_image_scaled(df):

        tmp = to_image(df)

        # Rescale Input
        tmp *= params["scale"]

        # TODO: fixme to per-image instead of per-batch
        if params["rnd_scale"]:
            tmp *= random.gauss(1.0,params["rnd_scale"])

        # Lower cut-off
        if params["cutoff"]:
            tmp[tmp < params["cutoff"]] = 0


        if params["sum2"]:

            for i_img in range(len(tmp)):
                ssq = TCB.np.sum(tmp[i_img]**2)
                if ssq >0:
                    tmp[i_img] /= ssq            

            return tmp

        else:
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

        activ = lambda : TCB.Activation('relu')
        model = TCB.Sequential()

        channels = 1
        nclasses = 2

        for i_block in range(params["n_blocks"]):
            for i_conv_layer in range(params["n_conv_layers"]):

                if i_conv_layer == 0 and i_block ==0:
                    model.add(TCB.ZeroPadding2D(padding=(1, 1), input_shape=(1, 40, 40)))
                else:
                    model.add(TCB.ZeroPadding2D(padding=(1, 1)))

                model.add(TCB.Convolution2D(params["conv_nfeat"],
                                        params["conv_size" ], 
                                        params["conv_size" ]))
                model.add(activ())

                if params["conv_batchnorm"]:
                    model.add(TCB.BatchNormalization())

                if params["conv_dropout"] > 0.0:
                    model.add(TCB.Dropout(params["conv_dropout"]))

            if params["pool_size"] > 0 and (i_block < params["n_blocks"] -1):
                model.add(TCB.MaxPooling2D(pool_size=(params["pool_size"], params["pool_size"])))

            if params["block_dropout"] > 0.0:
                model.add(TCB.Dropout(params["block_dropout"]))

        model.add(TCB.Flatten())

        for i_dense_layer in range(params["n_dense_layers"]):
            model.add(TCB.Dense(params["n_dense_nodes"]))
            model.add(activ())    

            if params["dense_batchnorm"]:
                model.add(TCB.BatchNormalization())

            if params["dense_dropout"] > 0.0:
                model.add(TCB.Dropout(params["dense_dropout"]))


        model.add(TCB.Dense(nclasses))
        model.add(TCB.Activation('softmax'))

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
        TCB.Classifier(params["model_name"],
                   "keras",
                   params,
                   params["read_from_file"],
                   datagen_train_pixel,
                   datagen_test_pixel,               
                   model_2d(params),
                   image_fun = to_image_scaled,           
                   class_names = {0: "background", 1: "signal"},
                   inpath = "/scratch/snx3000/gregork/outputs",
               )
    ]

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
        
        if params["suffix"]:
            TCB.eval_single(clf, params["suffix"])
        else:
            TCB.eval_single(clf)

    return "Done"

