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
# LorentzLayer (LoLa)
#
###

class LoLa(Layer):
    """A Keras layer for working with Lorentz vectors.

    Input shape:
    (None, n_features, n_particles, n_layers)
    OR
    (None, n_features, n_particles)
    which is transformed to n_layers = 1
      
    Output shape:
    (None, n_features, n_particles, n_filters)
    """

    def __init__(self, 
                 debug                   = False,
                 add_total = True,
                 add_eye   = True,
                 n_out_particles = 30,
                 **kwargs):

        self.debug = debug

        self.add_total = add_total
        self.add_eye   = add_eye
        self.n_out_particles = n_out_particles

        self.diff = True
        self.regularize = False

        super(LoLa, self).__init__(**kwargs)


    def build(self, input_shape):
        """Prepare all the weights for training"""
        
        self.n_features  = input_shape[1]

        if self.debug:
            self.n_particles = 5
            self.batch_size = 2
        else:
            self.n_particles = input_shape[2]
        
        print ("We have n_features={0} / n_particles={1}".format(
            self.n_features, self.n_particles))

        self.w_Aij = self.add_weight(
            "w_Aij",
            shape=(self.n_particles, self.n_out_particles),
            initializer='uniform',
            trainable=True)
        
        
#        self.w = self.add_weight(
#            "w",
#            shape=(2,),
#            initializer='uniform',
#            trainable=True)

        if self.regularize:
            self.w_reg = self.add_weight(
                "w_reg",
                shape=(3,),
                initializer='uniform',
                trainable=True)


        self.metric = K.variable(np.array([[ -1, 0, 0, 0],
                                           [  0, 1, 0, 0],
                                           [  0, 0, 1, 0],
                                           [  0, 0, 0, 1]]))

        # and build the layer
        super(LoLa, self).build(input_shape)  

            
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
            
        # magic1: 111 000 000
        #         000 111 000
        #         000 000 111
        #
        # magic2: 100 100 100
        #         010 010 010
        #         001 001 001                
        magic1 = K.repeat_elements(K.eye(self.n_particles),self.n_particles,1)
        magic2 = K.tile(K.eye(self.n_particles),[1,self.n_particles])

        # dimension: p p'
        if self.diff:
            magic = magic1-magic2
        else:
            magic = magic1+magic2

        if self.debug:
            print ("magic:")
            print (K.eval(magic.shape))
            print (K.eval(magic))

        # b f p p'
        # x * magic  gives b f p^2, reshape to b f p p'
        diff_ij = K.reshape(K.expand_dims(K.dot(x, magic), -1), (x.shape[0], x.shape[1], x.shape[2], x.shape[2]))
    
        if self.debug:
            print ("diff_ij:")
            print (K.eval(diff_ij.shape))
            print (K.eval(diff_ij))

        # fold with the metric
        # b f p p' * f f' = b p p' f 
        prod = theano.tensor.tensordot(diff_ij, self.metric, axes=[1,0])

        if self.debug:
            print ("prod:")
            print (K.eval(prod.shape))
            print (K.eval(prod))

        # multiply out the feature dimension
        # b p p' f * b p'' p ''' f'
        # -> b p p' p'' p'''
        prod = theano.tensor.batched_tensordot(prod, diff_ij, axes=[3,1])

        if self.debug:
            print ("prod:")
            print (K.eval(prod.shape))
            print (K.eval(prod))

        # build the diagonal we care about
        # b p p' p'' p''' -> b p p'
        # TODO: understand why we need these axes for correct diagonal?
        dij_square = prod.diagonal(axis1=1,axis2=4).diagonal(axis1=1,axis2=2)
                
        if self.debug:
            print ("dij_square:")
            print (K.eval(dij_square.shape))
            print (K.eval(dij_square))

        dij_square = K.clip(K.abs(K.pow(dij_square,-1)),0,1)
            
#        if self.debug:
#            print ("dijminus:")
#            print (K.eval(dijminus.shape))
#            print (K.eval(dijminus))

        # b
        lead_pxs = x[:,1,0]
        lead_pys = x[:,2,0]
        lead_pts2 = K.expand_dims(K.expand_dims(K.pow(lead_pxs+lead_pys,2),-1),-1)
        
        lead_pts2 = K.tile(lead_pts2, [1,self.n_particles, self.n_particles])


        if self.debug:
            print ("lead_pts2:")
            print (K.eval(lead_pts2.shape))
            print (K.eval(lead_pts2))

        denom = K.clip(lead_pts2, 0.001, 100000.)


        if self.debug:
            print ("self.w_Aij:")
            print (K.eval(self.w_Aij))
            print (K.eval(self.w_Aij))

        term1 = dij_square

        if self.debug:
            print ("term1:")
            print (K.eval(term1.shape))
            print (K.eval(term1))

        poly = self.w_Aij # + self.w[0] *  term1 # + self.w[1] * pow(dij_square/denom,2)

        if self.debug:
            print ("poly:")
            print (K.eval(poly.shape))
            print (K.eval(poly))

        if self.regularize:        
            poly = K.map_fn( lambda x:K.switch(K.less(x,self.w_reg[0]),
                                               self.w_reg[1],
                                               self.w_reg[2]), poly)



        li = []
        
        if self.add_total:
            li.append( K.ones(shape=(self.n_particles, 1)))
        
        if self.add_eye:
            li.append( K.eye(self.n_particles))
        
        li.append(poly)
        

        poly = K.concatenate(li,axis = 1)


        # And multiply with the initial particle positions
        #ret = K.batch_dot(x,poly)
        ret = K.dot(x,poly)

        if self.debug:
            print ("ret:")
            print (K.eval(ret.shape))
            print (K.eval(ret))


        if self.debug:
            sys.exit()

        return ret

    def compute_output_shape(self, input_shape):

        total_out_particles = self.n_out_particles

        if self.add_total:
            total_out_particles += 1
            
        if self.add_eye:
            total_out_particles += input_shape[2]
        
        
        return (input_shape[0], input_shape[1], total_out_particles)

