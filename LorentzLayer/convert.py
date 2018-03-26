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

def theta(pi,pj):

    metric = K.variable(np.array([[ -1, 0, 0, 0],
                                  [  0, 1, 0, 0],
                                  [  0, 0, 1, 0],
                                  [  0, 0, 0, 1]]))
    
    diff = pi-pj
    
    minkowski = K.dot(K.dot(metric,K.transpose(diff)),diff)
    
    denom =  (K.variable(2) * pi[0] * pj[0])
    
    denom = K.maximum(0.001, denom)
    
    
    one_minus_cos = minkowski / denom

    theta = K.variable(2) * one_minus_cos

    # TODO: Careful, this is not theta yes. Some arccos and stuff is missing. Let's see if it works now
    
    return theta
    

def mass(pi):

    metric = K.variable(np.array([[ -1, 0, 0, 0],
                                  [  0, 1, 0, 0],
                                  [  0, 0, 1, 0],
                                  [  0, 0, 0, 1]]))
        
    return  K.dot(K.dot(metric,K.transpose(pi)),pi)

def pt(pi):


    return K.pow(pi[1],2) + K.pow(pi[2],2)
        

    
    
class Convert(Layer):

    def __init__(self,                  
                 batch_size,
                 debug                   = False,
                 **kwargs):
        """        
        """

        self.debug = debug
                
        self.nout = 3
        self.batch_size = batch_size
        
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
                                         
        self.n_four_vectors = self.n_particles * self.n_layers

        # and build the layer
        super(Convert, self).build(input_shape)  
        

    def call(self, x):
        """Build the actual logic."""

        # (None, n_features, n_particles) -> (None, n_features, n_particles, n_layers)

        if self.debug:                     
            x= K.variable(np.array([[[ 229.46118164,  132.46817017,   26.43243217,    13.2313776,    5.75571156, 0],
                                     [-195.08522034, -113.19028473,  -22.73009872,  -10.31623554,   -4.25184822, 0],
                                     [-114.19178772,  -65.08143616,  -12.34527397,   -8.04754353,   -2.59461427, 0],
                                      [ -39.42618179,  -22.36474037,   -5.44153976,   -1.97019398,   -2.88409066, 0]]]))

            x = K.expand_dims(x, axis=-1)
        
        batches = []
        
        for ib in range(self.batch_size):

            #thetas = K.stack([theta(x[ib,:,0,0],x[ib,:,i,0]) for i in range(self.n_four_vectors)], axis=0)
            masses = K.stack([mass(x[ib,:,i,0]) for i in range(self.n_four_vectors)], axis=0)
            Es = K.stack([x[ib,0,i,0] for i in range(self.n_four_vectors)], axis=0)
            #EsNorm = K.stack([x[ib,0,i,0]/x[ib,0,0,0] for i in range(self.n_four_vectors)], axis=0)

            pts     = K.stack([pt(x[ib,:,i,0]) for i in range(self.n_four_vectors)], axis=0)
            #ptsNorm = K.stack([pt(x[ib,:,i,0])/pt(x[ib,:,0,0]) for i in range(self.n_four_vectors)], axis=0)

            trans = K.stack( [masses,Es, pts], axis=1)

            batches.append(trans)
            
        results = K.expand_dims(K.stack(batches,axis=0),-1)
            
        if self.debug:                    
            print("Results:")
            print(K.eval(results.shape))        
            print(K.eval(results))        

        return results


    def compute_output_shape(self, input_shape):
        
        print input_shape
        
        ret = (input_shape[0], self.nout, input_shape[2], self.n_layers)
        return ret

