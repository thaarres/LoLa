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
from keras.layers.core import Reshape, Dropout
from keras.models import model_from_yaml

print("Imported keras")

sys.path.append("../LorentzLayer")

from cola import CoLa
from lola import LoLa
from sola import SoLa
#from ala import ALa

#
# Prepare Jet Image
#

def to_image_2d(df):
    foo =  np.expand_dims(np.expand_dims(df[ ["c{0}".format(i) for i in range(40*40)]], axis=-1).reshape(-1,40,40), axis=1)        
    return foo

#
# Prepare Constituents
#

def to_constit(df, n_constit, n_features):

    brs = []

    if n_features == 4:
        feat_list =  ["E","PX","PY","PZ"] 
    elif n_features == 5:
        feat_list =  ["E","PX","PY","PZ","C"] 
    elif n_features == 8:
        feat_list =  ["E","PX","PY","PZ","C","VX", "VY", "VZ"] 

    brs += ["{0}_{1}".format(feature,constit) for feature in feat_list for constit in range(n_constit)]

    ret = np.expand_dims(df[brs],axis=-1).reshape(-1, n_features, n_constit)
    

    ret = ret/500.

    if n_features == 5:
        ret[:,4,:] = ret[:,4,:] * 500.
        ret[:,4,:] = pow(ret[:,4,:],2)

    if n_features == 8:
        ret[:,4,:] = ret[:,4,:] * 500.
        ret[:,4,:] = pow(ret[:,4,:],2)
        ret[:,5,:] = ret[:,5,:] * 500.
        ret[:,6,:] = ret[:,6,:] * 500.
        ret[:,7,:] = ret[:,7,:] * 500.

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

    model.add(CoLa(input_shape = (params["n_features"], params["n_constit"]),
                   add_total = True,
                   add_eye   = True,
                   n_out_particles = 15))

#    model.add(ALa(debug = False, threshold = 3/500.))

    model.add(LoLa(
        train_metric = False,
        es  = 0,
        xs  = 0,
        ys  = 0,
        zs  = 0,                 
        ms  = 1,                 
        pts = 1,                 
        n_train_es  = 1,
        n_train_ms  = 0,
        n_train_pts = 0,        
        n_train_sum_dijs   = 2,
        n_train_min_dijs   = 2))

    model.add(Flatten())

    model.add(Dense(100))
    model.add(Activation('relu'))

    model.add(Dense(50))
    model.add(Activation('relu'))

    model.add(Dense(params["n_classes"], activation='softmax'))

    return model
