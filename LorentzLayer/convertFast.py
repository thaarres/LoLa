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
# Convert
#
###

def PT(pi):
    E  = pi[:,0]
    pX = pi[:,1]
    pY = pi[:,2]
    pZ = pi[:,3]

    pT2 = K.pow(pX,2)+K.pow(pY,2)

    mass = E*E - pX*pX - pY*pY - pZ*pZ

    return K.stack([E, pT2, mass], axis=0)                




class Convert(Layer):

    def __init__(self,                  
                 debug                   = False,
                 **kwargs):
        """        
        """

        self.debug = debug
                
        self.nout = 3
        
        super(Convert, self).__init__(**kwargs)
 

    def build(self, input_shape):
        """Prepare all the weights for training"""
        
        self.n_features  = input_shape[1]
        self.n_particles = input_shape[2]

        if len(input_shape) == 3:
            # If we have
            # (None, n_features, n_particles)
            # as input, then infer that n_layers=1 
            # and expand along the last axis when appropriate
            self.n_layers    = 1
            self.need_expand = True
        else:
            self.n_layers = input_shape[3]
            self.need_expand = False

        print ("We have n_features={0} / n_particles={1} / n_layers={2}".format(
            self.n_features, self.n_particles, self.n_layers))
                                         
        # and build the layer
        super(Convert, self).build(input_shape)  
        

    def call(self, x):
        """Build the actual logic."""

        # (None, n_features, n_particles) -> (None, n_features, n_particles, n_layers)

        if self.debug:                     
            x= K.variable(np.array([[[ 229.46118164,  132.46817017,   26.43243217,    13.2313776,    5.75571156],
                                     [-195.08522034, -113.19028473,  -22.73009872,  -10.31623554,   -4.25184822],
                                     [-114.19178772,  -65.08143616,  -12.34527397,   -8.04754353,   -2.59461427],
                                     [ -39.42618179,  -22.36474037,   -5.44153976,   -1.97019398,   -2.88409066]]]))

                
        if self.need_expand:
            # (None, n_features, n_particles) -> (None, n_features, n_particles, 1)
            x = K.expand_dims(x, axis=-1)

        if self.debug:
            print ("x:")
            print (K.eval(x.shape))
            print (K.eval(x))            
        
        n_four_vectors = x.shape[0] * x.shape[2] * x.shape[3]
            
        x_perm = K.permute_dimensions(x, (0,3,2,1))
    
        if self.debug:
            print ("x_perm:")
            print (K.eval(x_perm.shape))
            print (K.eval(x_perm))            

        x_perm = K.reshape(x_perm, (n_four_vectors, 4))

        if self.debug:
            print ("x_perm:")
            print (K.eval(x_perm.shape))
            print (K.eval(x_perm))            


        trans = PT(x_perm)

        if self.debug:                    
            print("Trans:")
            print(K.eval(trans.shape))        
            print(K.eval(trans))        
        
        results = K.reshape(trans, (self.nout,x.shape[0],x.shape[2],x.shape[3]))

        if self.debug:                    
            print("Reshaped:")
            print(K.eval(results.shape))        
            print(K.eval(results))        

        results = K.permute_dimensions(results, (1,0,2,3))

        if self.debug:
            print ("permuted:")
            print (K.eval(results.shape))
            print (K.eval(results))            

        return results


    def compute_output_shape(self, input_shape):
        ret = (input_shape[0], self.nout, input_shape[2], self.n_layers)
        return ret

