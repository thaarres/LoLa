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
# SoLa
#
###


class SoLa(Layer):

    def __init__(self,                  
                 debug = False,
                 sort_by_feature = 0,
                 k_highest = 0,
                 **kwargs):
        """        
        """

        self.debug = debug

        self.sort_by_feature = sort_by_feature
        self.k_highest       = k_highest
        
        super(SoLa, self).__init__(**kwargs)
 

    def build(self, input_shape):
        """Prepare all the weights for training"""
        
        self.n_features  = input_shape[1]
        self.n_particles = input_shape[2]
        
        if self.debug:
            self.n_particles = 5

        print ("We have n_features={0} / n_particles={1}".format(
            self.n_features, self.n_particles))
                                         
        # and build the layer
        super(SoLa, self).build(input_shape)  
        

    def call(self, x):
        """Build the actual logic."""

        # (None, n_features, n_particles) -> (None, n_features, n_particles)

        if self.debug:                     
            x= K.variable(np.array([[[ 229.46118164,  132.46817017,   26.43243217,    13.2313776,    5.75571156],
                                     [-195.08522034, -113.19028473,  -22.73009872,  -10.31623554,   -4.25184822],
                                     [-114.19178772,  -65.08143616,  -12.34527397,   -8.04754353,   -2.59461427],
                                     [ -39.42618179,  -22.36474037,   -5.44153976,   -1.97019398,   -2.88409066]],
                                    [[ 129.,  135.,   26.,    15.,    7.],
                                     [-105., -114.,  -20.,  -10.,   -6.],
                                     [-100.,  -60.,  -10.,   -8.,   -1.],
                                     [ -32.,  -20.,   -5.,   -1.,   -2.]]]))


        ind = theano.tensor.argsort(x, axis = -1)
        
        if self.k_highest:
            kh = -1 * self.k_highest
            sorted_ind = ind[:,:,kh:]
        else:
            sorted_ind = ind

        sorted_ind = K.repeat(sorted_ind[:,self.sort_by_feature,:],self.n_features)

        dim0, dim1, dim2 = sorted_ind.shape

        indices_dim0 = theano.tensor.arange(dim0).repeat(dim1 * dim2)
        indices_dim1 = theano.tensor.arange(dim1).repeat(dim2).reshape((dim1*dim2, 1)).repeat(dim0, axis=1).T.flatten()

        results = x[indices_dim0, indices_dim1, sorted_ind.flatten()].reshape(sorted_ind.shape)
        
        if self.debug:
            sys.exit()
        
        return results

    def compute_output_shape(self, input_shape):
        return input_shape

