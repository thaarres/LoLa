import pdb
import sys

import numpy as np

from keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.utils import np_utils, generic_utils
from keras.layers.advanced_activations import PReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, ZeroPadding3D
from keras.layers.core import Reshape
from keras.models import model_from_yaml

print("Imported keras")

sys.path.append("../LorentzLayer")
from lola import LoLa
from convert import Convert


#
# Prepare Jet Image
#

def to_image_2d(df):
    foo =  np.expand_dims(np.expand_dims(df[ ["c{0}".format(i) for i in range(40*40)]], axis=-1).reshape(-1,40,40), axis=1)        
    return foo

#
# Prepare Constituents
#

def to_constit(df, n_constit):

    brs = []
    brs += ["{0}_{1}".format(feature,constit) for feature in ["E","PX","PY","PZ"] for constit in range(n_constit)]

    ret = np.expand_dims(df[brs],axis=-1).reshape(-1, 4, n_constit)
    
    # Reasonable jet scale for ML to work with
    ret = ret/500.

    return ret

#
# 2D ConvNet
#

def model_2d(params):

    activ = lambda : Activation('relu')
    model = Sequential()

    nclasses = params["n_classes"]

    for i_block in range(params["n_blocks"]):
        for i_conv_layer in range(params["n_conv_layers"]):

            if i_conv_layer == 0 and i_block ==0:
                model.add(Conv2D(params["conv_nfeat"],
                                            (params["conv_size" ], params["conv_size" ]),
                                            padding='same',
                                            input_shape=(1, 40, 40)))
            else:
                model.add(Conv2D(params["conv_nfeat"],
                                            (params["conv_size" ], params["conv_size" ]),
                                            padding='same'))


            model.add(activ())

            if params["conv_batchnorm"]:
                model.add(BatchNormalization())

            if params["conv_dropout"] > 0.0:
                model.add(Dropout(params["conv_dropout"]))


        if params["pool_size"] > 0 and (i_block < params["n_blocks"] -1):
            if params["pool_type"] == "max":
                model.add(MaxPooling2D(pool_size=(params["pool_size"], params["pool_size"])))
            elif params["pool_type"] == "avg":
                model.add(AveragePooling2D(pool_size=(params["pool_size"], params["pool_size"])))

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

#
# FCN
#

def model_fcn(params):

    activ = lambda : Activation('relu')
    model = Sequential()
    
    model.add(Flatten(input_shape=(4,params["n_constit"])))

    model.add(Dense(20, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(20, activation='relu'))

    model.add(Dense(params["n_classes"]))
    model.add(Activation('softmax'))

    return model

#
# LoLa
#

def model_lola(params):

    model = Sequential()

    debug = False

    model.add(LoLa(input_shape             = (4, params["n_constit"]),
                   train_poly              = params["train_poly"],
                   train_offset            = params["train_offset"],
                   train_metric            = params["train_metric"],
                   train_minmax            = params["train_minmax"],
                   n_filters               = params["lola_filters"],
                   regularize_weight       = params["regularize_weight"],
                   train_regularize_weight = params["train_regularize_weight"],
                   train_regularize_weight_target = params["train_regularize_weight_target"],                                                                                        
                   do_mult_p               = params["do_mult_p"],
                   mult_p                  = params["mult_p"],
                   use_angular_dr          = params["use_angular_dr"],                   
                   take_diff               = params["take_diff"],                   
                   debug                   = debug,                               
                   
               ))

    if params["n_lolas"] > 1:
        for _ in range(params["n_lolas"]-1):
            model.add(LoLa(
                train_poly              = params["train_poly"],
                train_offset            = params["train_offset"],
                train_metric            = params["train_metric"],
                train_minmax            = params["train_minmax"],
                n_filters               = params["lola_filters"],
                regularize_weight       = params["regularize_weight"],
                train_regularize_weight = params["train_regularize_weight"],
                train_regularize_weight_target = params["train_regularize_weight_target"],
                do_mult_p               = params["do_mult_p"],
                mult_p                  = params["mult_p"],
                use_angular_dr          = params["use_angular_dr"],                   
                take_diff               = params["take_diff"],                   
                debug                   = debug))

    model.add(Convert())            
 
    model.add(Flatten())

    model.add(Dense(20))
    model.add(Activation('relu'))

    model.add(Dense(20))
    model.add(Activation('relu'))

    model.add(Dense(params["n_classes"], activation='softmax'))

    return model
