###
#
# Imports
#
###

import pdb
import sys

# Needed for our architecture
sys.setrecursionlimit(10000)

import numpy as np

import theano

from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers

###
#
# CombinationLayer (CoLa)
#
###

class CoLa(Layer):
    """A Keras layer for creating linear combinations of Lorentz vectors."""


    def __init__(self, 
                 debug           = False,
                 add_total       = True,
                 add_eye         = True,
                 n_out_particles = 30,
                 **kwargs):

        self.debug = debug

        self.add_total       = add_total
        self.add_eye         = add_eye
        self.n_out_particles = n_out_particles

        super(CoLa, self).__init__(**kwargs)


    def build(self, input_shape):
        """Prepare all the weights for training"""
        
        self.n_features  = input_shape[1]

        if self.debug:
            self.n_particles = 5
            self.batch_size  = 2
        else:
            self.n_particles = input_shape[2]
        
        print ("We have n_features={0} / n_particles={1}".format(
            self.n_features, self.n_particles))

        
        self.w_Aij = self.add_weight(
            "w_Aij",
            shape=(self.n_particles, self.n_out_particles),
            initializer='uniform',
            trainable=True)


        self.total_out_particles = self.n_out_particles

        if self.add_total:
            self.total_out_particles += 1
            
        if self.add_eye:
            self.total_out_particles += self.n_particles

        
        # and build the layer
        super(CoLa, self).build(input_shape)  

            
    def call(self, x):
        """Build the actual logic."""

        # Our input has the shape
        # (batch, features, particles)        
        # Short: bfp

        if self.debug:                     
            x= K.variable(np.array([[[ 229.46118164,  132.46817017,   26.43243217,    13.2313776,    5.75571156],
                                     [-195.08522034, -113.19028473,  -22.73009872,  -10.31623554,   -4.25184822],
                                     [-114.19178772,  -65.08143616,  -12.34527397,   -8.04754353,   -2.59461427],
                                     [ -39.42618179,  -22.36474037,   -5.44153976,   -1.97019398,   -2.88409066],
                                     [ 1           ,             1,             0,             0,             1],
                                     [ 1.2         ,           -1.,             0,             0,             1],
                                     [ 1.2         ,           -2.,             0,             0,             1],
                                     [ 1.2         ,           -3.,             0,             0,             1],                                     
                                 ],
                                    [[ 129.,  135.,   26.,    15.,    7.],
                                     [-105., -114.,  -20.,  -10.,   -6.],
                                     [-100.,  -60.,  -10.,   -8.,   -1.],
                                     [ -32.,  -20.,   -5.,   -1.,   -2.],
                                     [   0.,    0.,    0.,    0.,    0.],
                                     [   0.,    0.,    0.,    0.,    0.],
                                     [   0.,    0.,    0.,    0.,    0.],
                                     [   1.,    1.,    1.,    1.,    1.],
                                 ]]))

        if self.debug:
            print ("x:")
            print (K.eval(x.shape))
            print (K.eval(x))

        li = []
        
        if self.add_total:
            li.append( K.ones(shape=(self.n_particles, 1)))
        
        if self.add_eye:
            li.append( K.eye(self.n_particles))
        
        li.append(self.w_Aij)
        
        Cij = K.concatenate(li,axis = 1)

        # And multiply with the initial particle positions
        ret = K.dot(x,Cij)

#         if self.debug:
#             print ("ret:")
#             print (K.eval(ret.shape))
#             print (K.eval(ret))
# 
#         # Unroll the matrix
#         # [E1, E2, E3, px1, px2, px3, ...., vz1, vz2, vz3]
#         # (assuming three particles)
#         ret = K.reshape(ret, (-1, self.total_out_particles * self.n_features))
# 
#         if self.debug:
#             print ("Cij:")
#             print (K.eval(Cij.shape))
#             print (K.eval(Cij))
#         
#         # Sum over the input particles
#         Cij_sum = K.sum(Cij, axis = 0)
# 
#         if self.debug:
#             print ("Cij_sum:")
#             print (K.eval(Cij_sum.shape))
#             print (K.eval(Cij_sum))
# 
#         Cij_sum_signs = K.sign(Cij_sum)
# 
#         Cij_sum_abs = K.maximum(0.001, K.abs(Cij_sum))
# 
#         denom = 1./(Cij_sum_signs * Cij_sum_abs)
# 
#     
#         oners = K.concatenate([K.ones((self.total_out_particles * 5 )), denom, denom, denom])
# 
#         oners = theano.tensor.nlinalg.diag(oners)
# 
#         if self.debug:
#             print ("denom:")
#             print (K.eval(denom.shape))
#             print (K.eval(denom))
# 
# 
#         if self.debug:
#             print ("oners:")
#             print (K.eval(oners.shape))
#             print (K.eval(oners))
# 
#         
#         if self.debug:
#             print ("ret:")
#             print (K.eval(ret.shape))
#             print (K.eval(ret))
# 
#         ret = K.dot(ret, oners)
# 
#         if self.debug:
#             print ("ret:")
#             print (K.eval(ret.shape))
#             print (K.eval(ret))
# 
# 
#         ret = K.reshape(ret, (-1, self.n_features, self.total_out_particles))
# 
#         if self.debug:
#             print ("ret:")
#             print (K.eval(ret.shape))
#             print (K.eval(ret))
# 
        
        if self.debug:
            sys.exit()

        return ret

    def compute_output_shape(self, input_shape):
                
        return (input_shape[0], input_shape[1], self.total_out_particles)

