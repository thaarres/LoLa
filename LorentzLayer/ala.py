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
# Activation layer (ALa)
#
###

class ALa(Layer):
    """A Keras layer for creating linear combinations of Lorentz vectors."""


    def __init__(self, 
                 debug           = False,                 
                 threshold       = 5/500.,
                 **kwargs):

        self.debug = debug        
        self.threshold = threshold

        super(ALa, self).__init__(**kwargs)


    def build(self, input_shape):
        
        self.n_features  = input_shape[1]

        if self.debug:
            self.n_particles = 5
            self.batch_size  = 2
        else:
            self.n_particles = input_shape[2]
        
        print ("We have n_features={0} / n_particles={1}".format(
            self.n_features, self.n_particles))

        # and build the layer
        super(ALa, self).build(input_shape)  

            
    def call(self, x):
        """Build the actual logic."""

        # Our input has the shape
        # (batch, features, particles)        
        # Short: bfp

        if self.debug:                     
            x= K.variable(np.array([[[ 229.46118164,  132.46817017,   26.43243217,    13.2313776,    5.75571156],
                                     [-195.08522034, -113.19028473,  -22.73009872,  -10.31623554,   -4.25184822],
                                     [-114.19178772,  -65.08143616,  -12.34527397,   -8.04754353,   -2.59461427],
                                     [ -39.42618179,  -22.36474037,   -5.44153976,   -1.97019398,   -2.88409066]],
                                    [[ 129.,  135.,   26.,    15.,    7.],
                                     [-105., -114.,  -20.,  -10.,   -6.],
                                     [-100.,  -60.,  -10.,   -8.,   -1.],
                                     [ -32.,  -20.,   -5.,   -1.,   -2.]]]))

        if self.debug:
            print ("x:")
            print (K.eval(x.shape))
            print (K.eval(x))

        li = []
        
#        # And multiply with the initial particle positions
#        ret = K.dot(x,Cij)

        # Element wise square of x
        # bfp
        x2 = pow(x,2)

        # bp
        Pts = K.abs(K.sqrt(x2[:,1,:] + x2[:,2,:]))

        if self.debug:
            print ("Pts:")
            print (K.eval(Pts.shape))
            print (K.eval(Pts))


        # subtract threshold
        Pts = Pts - self.threshold

        if self.debug:
            print ("Pts:")
            print (K.eval(Pts.shape))
            print (K.eval(Pts))

        # Apply heaviside step function
        stepped = (theano.tensor.sgn(Pts) + 1.000001)/2.000001

        if self.debug:
            print ("stepped:")
            print (K.eval(stepped.shape))
            print (K.eval(stepped))


        stepped = K.expand_dims(stepped, axis=1)


        if self.debug:
            print ("stepped:")
            print (K.eval(stepped.shape))
            print (K.eval(stepped))


        stepped = K.repeat_elements(stepped, 4, 1)


        if self.debug:
            print ("stepped:")
            print (K.eval(stepped.shape))
            print (K.eval(stepped))
            
        ret = x * stepped

        if self.debug:
            print ("ret:")
            print (K.eval(ret.shape))
            print (K.eval(ret))

        if self.debug:
            sys.exit()

        return ret
