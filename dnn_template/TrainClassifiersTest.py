import faulthandler

faulthandler.enable()

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
        
        "inputs" : "hd",

        # Parameters for 2d convolutional architecture    
        "n_blocks"        : 1,    
        "n_conv_layers"   : 1,        
        "conv_nfeat"      : 1,
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



    name = "v39_"
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
    
    pixel_brs = []

    if params["inputs"] == "hd":
        pixel_brs += ["hd_{0}".format(i) for i in range(15*15)]
    elif params["inputs"] == "em" :
        pixel_brs += ["em_{0}".format(i) for i in range(15*15)]
    elif params["inputs"] == "tk":
        pixel_brs += ["tk_{0}".format(i) for i in range(15*15)]
    elif params["inputs"] == "3d":
        pixel_brs += ["hd_{0}".format(i) for i in range(15*15)]
        pixel_brs += ["em_{0}".format(i) for i in range(15*15)]
        pixel_brs += ["tk_{0}".format(i) for i in range(15*15)]

    # Reading H5FS
    infname_train = TCB.os.path.join(params["input_path"], "train-qg_v0.h5")
    infname_test  = TCB.os.path.join(params["input_path"], "test-qg_v0.h5")


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

    datagen_train_pixel = TCB.datagen_batch_h5(pixel_brs, infname_train, batch_size=params["batch_size"])
    datagen_test_pixel  = TCB.datagen_batch_h5(pixel_brs, infname_test, batch_size=params["batch_size"])

    for foo in range(n_train_samples/params["batch_size"]):
        print (foo, datagen_train_pixel.next())

main({"input_path" : "/scratch/snx3000/gregork/",
          "model_name" : "simple_XX",        
          "inputs"     : "3d",})

