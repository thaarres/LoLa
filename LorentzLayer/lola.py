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

import tensorflow as tf
#import theano #theano

from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers

###
#
# LoLa
#
###

class LoLa(Layer):

    def __init__(self,                  
                 debug        = False,
                 train_metric = False,

                 n_divide = 30,

                 es  = 0,
                 xs  = 0,
                 ys  = 0,
                 zs  = 0,

                 cs  = 0,
                 vxs = 0,
                 vys = 0,
                 vzs = 0,

                 ms  = 1,                 
                 pts = 1,
                 dls = 0, 
                 
                 n_train_es  = 1,
                 n_train_ms  = 0,
                 n_train_pts = 0,

                 n_train_sum_dijs   = 2,
                 n_train_min_dijs   = 2,
    
                 **kwargs):
        """        
        """

        self.debug = debug

        self.n_divide = n_divide
                
        self.es  = es 
        self.xs  = xs 
        self.ys  = ys 
        self.zs  = zs 

        self.cs  = cs 
        self.vxs = vxs
        self.vys = vys
        self.vzs = vzs

        self.ms  = ms 
        self.pts = pts
        self.dls = dls

        self.n_train_es  = n_train_es 
        self.n_train_ms  = n_train_ms 
        self.n_train_pts = n_train_pts

        self.n_train_sum_dijs   = n_train_sum_dijs  
        self.n_train_min_dijs   = n_train_min_dijs  
        
        self.train_metric = train_metric

        self.total_trains = (self.n_train_es + 
                             self.n_train_ms + 
                             self.n_train_pts + 
                             self.n_train_sum_dijs + 
                             self.n_train_min_dijs                              
                         )

        self.do_blocks = 0

        self.nout = (int(self.xs)  + 
                     int(self.ys)  + 
                     int(self.zs)  + 
                     int(self.cs)  + 
                     int(self.vxs) +
                     int(self.vys) + 
                     int(self.vzs) +
                     int(self.ms)  + 
                     int(self.es)  + 
                     int(self.pts) + 
                     self.total_trains
                 )

        super(LoLa, self).__init__(**kwargs)
 

    def build(self, input_shape):
        """Prepare all the weights for training"""
        
        self.n_features  = input_shape[1]
        self.n_particles = input_shape[2]
        
        # first N particles: initial inputs
        # n_particles-N: trained outputs
        self.N = 30
        
        if self.debug:
            self.n_particles = 5

        print ("We have n_features={0} / n_particles={1}".format(
            self.n_features, self.n_particles))
                 
        if self.total_trains > 0:

            if self.do_blocks == 1:
                self.w = self.add_weight(
                    "w",
                    shape=(self.total_trains, self.n_divide, self.n_divide),
                    initializer='uniform',
                    trainable=True)
            else:
                self.w = self.add_weight(
                    "w",
                    shape=(self.total_trains, self.n_particles, self.n_particles),
                    initializer='uniform',
                    trainable=True)

#        self.spatm = self.add_weight(
#            "spatm",
#            shape=(1, 3),
#            initializer='uniform',
#            trainable=True)

                
        if self.train_metric:
            self.m = self.add_weight(
                "m",
                shape=(1, self.n_features),
                initializer='uniform',
                trainable=True)

        # and build the layer
        super(LoLa, self).build(input_shape)  
        

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

        weight_index = 0
        out_features = []

        if self.train_metric:
            metric = self.m[0,:]
        else:
            # Build the Minkowski metric. Ignore extra features if present
            metric_vector = [ -1., 1., 1., 1.]
            if self.n_features > 4:
                metric_vector.extend( [0.] * (self.n_features - 4))
            # metric = K.variable(np.array(metric_vector)) #theano
            metric = tf.constant(metric_vector)

        #spatial_metric = K.variable(np.array([0.,0.,0.,0.,0.,1.,1.,1.]))
        #spatial_metric = K.concatenate([K.variable(np.array([0.,0.,0.,0.,0.])),self.spatm[0,:]])
        
        if self.do_blocks == 1:
            self.w = K.concatenate([self.w, K.zeros(shape=(self.total_trains, self.n_divide, self.n_particles - self.n_divide))], axis = -1)
            self.w = K.concatenate([self.w, K.zeros(shape=(self.total_trains, self.n_particles - self.n_divide, self.n_particles))], axis=1)
 
        if self.debug: 
            print("weights")
            print(K.eval(self.w.shape))      
            print(K.eval(self.w))

        # Our input is of the form 
        # (b,f,p)
        # -> (batch_size, features, particles)
            
        # Let's build a few helpful matrices

        # All the individual dimensions
        # bp
        Es = x[:,0,:]        
        Xs = x[:,1,:]        
        Ys = x[:,2,:]        
        Zs = x[:,3,:]        

#        Cs = x[:,4,:]        
#        Vxs = x[:,5,:]        
#        Vys = x[:,6,:]        
#        Vzs = x[:,7,:]        
        
        # Element wise square of x
        # bfp
        # x2 = pow(x,2) #theano
        x2 = tf.square(x) 

        # Mass^2 and transverse momentum^2
        # bp
        # Ms  = theano.tensor.tensordot(x2, metric, axes=[1,0]) #theano
        Ms  = tf.tensordot(x2, metric, axes=[[1], [0]])

        Pts = K.abs(K.sqrt(x2[:,1,:] + x2[:,2,:]))

        Pts2 = x2[:,1,:] + x2[:,2,:]

        #Vds = x2[:,5,:] +  x2[:,6,:]
        #Vds = theano.tensor.tensordot(x2, spatial_metric, axes=[1,0])


        if self.es:
            out_features.append(Es)
        if self.xs:
            out_features.append(Xs)
        if self.ys:
            out_features.append(Ys)
        if self.zs:
            out_features.append(Zs)

        if self.cs:
            out_features.append(Cs)
        if self.vxs:
            out_features.append(Vxs)
        if self.vys:
            out_features.append(Vys)
        if self.vzs:
            out_features.append(Vzs)

        if self.ms:
            out_features.append(Ms)
        if self.pts:
            out_features.append(Pts)

        #out_features.append(Vds)

        # difference to leading particle
        if self.dls:                           
            dl = pow(x-K.repeat_elements(K.expand_dims(x[:,:,0],-1), self.n_particles, -1),2)
            dl = theano.tensor.tensordot(dl, metric, axes=[1,0])
            out_features.append(dl)

        for i in range(self.n_train_es):
            # out_features.append(theano.tensor.tensordot(Es, self.w[weight_index,:,:], axes=[1,0])) #theano
            out_features.append(tf.tensordot(Es, self.w[weight_index,:,:], axes=[[1], [0]]))
            weight_index += 1

        for i in range(self.n_train_ms):
            # out_features.append(theano.tensor.tensordot(Ms, self.w[weight_index,:,:], axes=[1,0]))
            out_features.append(tf.tensordot(Ms, self.w[weight_index,:,:], axes=[[1], [0]]))
            weight_index += 1

        for i in range(self.n_train_pts):
            # out_features.append(theano.tensor.tensordot(Pts, self.w[weight_index,:,:], axes=[1,0]))
            out_features.append(tf.tensordot(Pts, self.w[weight_index,:,:], axes=[[1], [0]]))
            weight_index += 1


        # Helper tensor for building sums/differences
        # magic1: 111 000 000
        #         000 111 000
        #         000 000 111
        #
        # magic2: 100 100 100
        #         010 010 010
        #         001 001 001                
        magic1 = K.repeat_elements(K.eye(self.n_particles),self.n_particles,1)
        magic2 = K.tile(K.eye(self.n_particles),[1,self.n_particles])
        magic_diff = magic1-magic2


        # Build d_ij^2 = (k_i - k_j)^mu (k_i - k_j)^nu eta_mu_nu

        # b f p p'
        # x * magic  gives b f p^2, reshape to b f p p'
        # d2_ij = K.reshape(K.expand_dims(K.dot(x, magic_diff), -1), (x.shape[0], x.shape[1], x.shape[2], x.shape[2])) #theano
        d2_ij   = K.reshape(K.expand_dims(K.dot(x, magic_diff), -1), (tf.shape(x)[0],tf.shape(x)[1],tf.shape(x)[2],tf.shape(x)[2]))
        
        # elements squared
        d2_ij = K.pow(d2_ij, 2)

        # fold with the metric
        # b f p p' * f  = b p p'  

        for i in range(self.n_train_sum_dijs):

            if self.debug:
                print("metric:")
                print(K.eval(metric.shape))      
                print(K.eval(metric))

            if self.debug:
                print("d2_ij:")
                print(K.eval(d2_ij.shape))
                print(K.eval(d2_ij))
            
            #m_d2_ij = theano.tensor.tensordot(d2_ij, metric, axes=[1,0]) #theano
            m_d2_ij = tf.tensordot(d2_ij, metric, axes=[[1],[0]])

            # out_features.append(K.sum(theano.tensor.tensordot(m_d2_ij, self.w[weight_index,:,:], axes=[1,0]), axis=2))
            out_features.append(K.sum(tf.tensordot(m_d2_ij, self.w[weight_index,:,:], axes=[[1],[0]]), axis=2))
            weight_index += 1





        if self.debug:
            print("Done with d2_ijs")


        # Build m_ij^2 = (k_i + k_j)^mu (k_i + k_j)^nu eta_mu_nu

        # b f p p'
        # x * magic  gives b f p^2, reshape to b f p p'
        #m2_ij = K.reshape(K.expand_dims(K.dot(x, magic_diff), -1), (x.shape[0], x.shape[1], x.shape[2], x.shape[2])) #theano
        m2_ij = K.reshape(K.expand_dims(K.dot(x, magic_diff), -1), (tf.shape(x)[0],tf.shape(x)[1],tf.shape(x)[2],tf.shape(x)[2]))
        
        # elements squared
        m2_ij = K.pow(m2_ij, 2)

        # fold with the metric
        # b f p p' * f  = b p p'  

        for i in range(self.n_train_min_dijs):

            # m_m2_ij = theano.tensor.tensordot(m2_ij, metric, axes=[1,0]) #theano
            m_m2_ij = tf.tensordot(m2_ij, metric, axes=[[1],[0]])
            
            # out_features.append(K.min(theano.tensor.tensordot(m_m2_ij, self.w[weight_index,:,:], axes=[1,0]), axis=2)) #theano
            out_features.append(K.min(tf.tensordot(m_m2_ij, self.w[weight_index,:,:], axes=[[1],[0]]), axis=2))
            weight_index += 1

        #vd_m2_ij = theano.tensor.tensordot(m2_ij, spatial_metric, axes=[1,0])
        #out_features.append(K.min(theano.tensor.tensordot(vd_m2_ij, self.w[weight_index,:,:], axes=[1,0]), axis=2))
        #weight_index += 1

        #vd_m2_ij2 = theano.tensor.tensordot(m2_ij, spatial_metric, axes=[1,0])
        #out_features.append(K.min(theano.tensor.tensordot(vd_m2_ij2, self.w[weight_index,:,:], axes=[1,0]), axis=2))
        #weight_index += 1


            
        if self.debug:
            print("done with m2_ij")
 

        # TODO: Also enable these..
        Pts_over_lead = Pts/K.repeat_elements(K.expand_dims(Pts[:,0], axis=-1),self.n_particles, -1)
        Es_over_lead  = Es/K.repeat_elements(K.expand_dims(Es[:,0], axis=-1),self.n_particles, -1)

                
        results = K.stack(out_features, axis = 1)

        if self.debug:
            print ("results:")
            print (K.eval(results))
            print (K.eval(results.shape))

        
        if self.debug:
            sys.exit()
        
        return results


    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.nout, input_shape[2]) 

