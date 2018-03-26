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
# PreFil
#
###


class PreFil(Layer):

    def __init__(self,                  
                 debug                   = False,
                 train_metric = False,
                 **kwargs):
        """        
        """

        self.debug = debug

        self.n_extras = 2
        
        super(PreFil, self).__init__(**kwargs)
 

    def build(self, input_shape):
        """Prepare all the weights for training"""
        
        self.n_features  = input_shape[1]
        self.n_particles = input_shape[2]
        
        if self.debug:
            self.n_particles = 5

        print ("We have n_features={0} / n_particles={1}".format(
            self.n_features, self.n_particles))
                                         
        self.t = self.add_weight(
            "t",
            shape=(1, self.n_extras),
            initializer='uniform',
            trainable=True)

        # and build the layer
        super(PreFil, self).build(input_shape)  
        

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

        Es = x[:,0,:]        
        
        extras = [x]
        
        for i in range(self.n_extras):

            extras.append(K.expand_dims(K.map_fn( lambda x:K.switch(K.less(x,self.t[0,i]),
                                                                    1.,
                                                                    0.), Es), 1))
 

            if self.debug:
                print("extra:")
                print(K.eval(extras[-1].shape))
                print(K.eval(extras[-1]))

    
        results = K.concatenate(extras, axis = 1)

        if self.debug:
            print ("results:")
            print (K.eval(results))
            print (K.eval(results.shape))

        
        if self.debug:
            sys.exit()
        
        return results


    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] + self.n_extras, input_shape[2]) 

