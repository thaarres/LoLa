def main(kwargs):

    print(kwargs)

    import TrainClassifiersBase as TCB

    ########################################
    # Configuration
    ########################################

    default_params = {        

        "model_name" : "NNXd_et_5deg_sample_v7_v37",
        
        "suffix" : "",

        # Input path
        "input_path" : "/scratch/snx3000/gregork/",

        # False: Train; True: read weights file 
        "read_from_file" : False,
        
        "inputs" : "2d",

        # Parameters for 2d convolutional architecture    
        "n_blocks"        : 1,    
        "n_conv_layers"   : 2,        
        "conv_nfeat"      : 4,
        "conv_size"       : 2,
        "conv_batchnorm"  : 0,
        "pool_size"       : 0,
        "n_dense_layers"  : 1,
        "n_dense_nodes"   : 20,
        "dense_batchnorm" : 0,

        "conv_dropout"    : 0.0,
        "block_dropout"   : 0.0,
        "dense_dropout"   : 0.0,

        "pool_type"       : "max",

        # Image pre-processing
        "cutoff"          : 0.0,
        "scale"           : 1.0,
        "rnd_scale"       : 0.0,
        "sum2"            : 0,

        # Common parameters
        "batch_size"        : 2000,
        "lr"                : 0.1,
        "decay"             : 0.,
        "momentum"          : 0.,            
        "nb_epoch"          : 20,
        "samples_per_epoch" : None, # later filled from input files
    }


    name = "w_"
    for k,v in kwargs.items():
        name += "{0}_{1}_".format(k,v)
    default_params["model_name"]=name        
        
    colors = ['black', 'red','blue','green','orange','green','magenta']

    ########################################
    # Read in parameters
    ########################################

    params = {}
    for param in default_params.keys():

        if param in kwargs.keys():
            cls = default_params[param].__class__
            value = cls(kwargs[param])
            params[param] = value
        else:
            params[param] = default_params[param]

    print("Parameters are:")
    for k,v in params.items():
        print("{0}={1}".format(k,v))

    tot_pool = params["pool_size"]**params["n_blocks"]
 
    if tot_pool > 0:
        if not ((40 % tot_pool == 0) and (tot_pool <= 40)):
            print("Total pool of {0} is too large. Exiting.".format(tot_pool))
            return 10.

    brs = ["is_singal_new"]

    pixel_brs = []

    if params["inputs"] == "2d":
        pixel_brs += ["img_{0}".format(i) for i in range(40*40)]

    # Reading H5FS
    infname_train = TCB.os.path.join(params["input_path"], "deepw-train-v1.h5")
    infname_test  = TCB.os.path.join(params["input_path"], "deepw-test-v1.h5")


    ########################################
    # H5FS: Count effective training samples
    ########################################

    store_train = TCB.pandas.HDFStore(infname_train)
    n_train_samples = int((store_train.get_storer('table').nrows/params["batch_size"]))*params["batch_size"]

    store_test = TCB.pandas.HDFStore(infname_test)
    n_test_samples = int((store_test.get_storer('table').nrows/params["batch_size"]))*params["batch_size"]

    print("Total number of training samples = ", n_train_samples)
    print("Total number of testing samples = ", n_test_samples)
    params["samples_per_epoch"] = n_train_samples
    params["samples_per_epoch_test"] = n_test_samples


    ########################################
    # Prepare data and scalers
    ########################################

    print(n_train_samples)

    datagen_train_pixel = TCB.datagen_batch_h5(brs+pixel_brs, infname_train, batch_size=params["batch_size"])
    datagen_test_pixel  = TCB.datagen_batch_h5(brs+pixel_brs, infname_test, batch_size=params["batch_size"])

    def to_image_2d(df):
        foo =  TCB.np.expand_dims(TCB.np.expand_dims(df[ ["img_{0}".format(i) for i in range(40*40)]], axis=-1).reshape(-1,40,40), axis=1)        
        return foo

    def model_2d(params):

        activ = lambda : TCB.Activation('relu')
        model = TCB.Sequential()

        nclasses = 2

        for i_block in range(params["n_blocks"]):
            for i_conv_layer in range(params["n_conv_layers"]):

                if i_conv_layer == 0 and i_block ==0:
                    model.add(TCB.Conv2D(params["conv_nfeat"],
                                                (params["conv_size" ], params["conv_size" ]),
                                                padding='same',
                                                input_shape=(1, 40, 40)))
                else:
                    model.add(TCB.Conv2D(params["conv_nfeat"],
                                                (params["conv_size" ], params["conv_size" ]),
                                                padding='same'))
            
                
                model.add(activ())

                if params["conv_batchnorm"]:
                    model.add(TCB.BatchNormalization())

                if params["conv_dropout"] > 0.0:
                    model.add(TCB.Dropout(params["conv_dropout"]))

            
            if params["pool_size"] > 0 and (i_block < params["n_blocks"] -1):
                if params["pool_type"] == "max":
                    model.add(TCB.MaxPooling2D(pool_size=(params["pool_size"], params["pool_size"])))
                elif params["pool_type"] == "avg":
                    model.add(TCB.AveragePooling2D(pool_size=(params["pool_size"], params["pool_size"])))

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



    the_model = model_2d
    the_image_fun = to_image_2d

    classifiers = [
        TCB.Classifier(params["model_name"],
                   "keras",
                   params,
                   params["read_from_file"],
                   datagen_train_pixel,                   
                   datagen_test_pixel,               
                   the_model(params),
                   image_fun = the_image_fun,           
                   class_names = {0: "background", 1: "signal"},
                   inpath = "/scratch/snx3000/gregork/outputs",
               )
    ]

    ########################################
    # Train/Load classifiers and make ROCs
    ########################################

    # Returns best val loss for keras training
    for clf in classifiers:
        clf.prepare()
                
        if params["suffix"]:
            ret = TCB.eval_single(clf, params["suffix"])
        else:
            ret = TCB.eval_single(clf)

        return ret
    

