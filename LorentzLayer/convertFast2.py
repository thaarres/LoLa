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


class Convert(Layer):

    def __init__(self,                  
                 debug                   = False,
                 train_metric = False,
                 **kwargs):
        """        
        """

        self.debug = debug
                
        
        self.ms  = 1
        self.es  = 0
        self.pts = 1

        self.n_train_es  = 1
        self.n_train_ms  = 0
        self.n_train_pts = 0

        self.n_train_a1s = 0
        self.n_train_a2s = 0

        self.n_train_dijs   = 2
        self.n_train_mijs   = 2
        self.n_train_cosijs = 0

        self.n_train_dijs_3d = 0
        self.n_train_mijs_3d = 0
        
        self.train_metric = train_metric
        self.n_metrics = self.n_train_dijs + self.n_train_mijs 

        self.total_trains = (self.n_train_es + 
                             self.n_train_ms + 
                             self.n_train_pts + 
                             self.n_train_a1s +
                             self.n_train_a2s +
                             self.n_train_dijs + 
                             self.n_train_mijs + 
                             self.n_train_dijs_3d + 
                             self.n_train_mijs_3d + 
                             self.n_train_cosijs
                         )
        
        self.nout = int(self.ms) + int(self.es) + int(self.pts) + self.total_trains

        super(Convert, self).__init__(**kwargs)
 

    def build(self, input_shape):
        """Prepare all the weights for training"""
        
        self.n_features  = input_shape[1]
        self.n_particles = input_shape[2]
        
        if self.debug:
            self.n_particles = 5

        print ("We have n_features={0} / n_particles={1}".format(
            self.n_features, self.n_particles))
                                         
        self.w = self.add_weight(
            "w",
            shape=(self.total_trains, self.n_particles, self.n_particles),
            initializer='uniform',
            trainable=True)

        if self.train_metric:
            self.m = self.add_weight(
                "m",
                shape=(self.n_metrics, self.n_features),
                initializer='uniform',
                trainable=True)


        #self.t = self.add_weight(
        #    "t",
        #    shape=(1,1),
        #    initializer='uniform',
        #    trainable=True)

        # and build the layer
        super(Convert, self).build(input_shape)  
        

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
        metric_index = 0
        out_features = []

        
        metric3 = K.variable(np.array([ 0., 1., 1., 1.]))

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

#        A1s = x[:,4,:]        
#        A2s = x[:,5,:]        

        # Element wise square of x
        # bfp
        x2 = pow(x,2)

        # Mass^2 and transverse momentum^2
        # bp
        Ms  = x2[:,0,:] - x2[:,1,:] - x2[:,2,:] - x2[:,3,:]
        Pts = K.abs(K.sqrt(x2[:,1,:] + x2[:,2,:]))
        

        #Pts = K.map_fn( lambda x:K.switch(K.less(x,self.t[0,0]),0,1), Es)

        if self.ms:
            out_features.append(Ms)
        if self.es:
            out_features.append(Es)
        if self.pts:
            out_features.append(Pts)

#        out_features.append(A1s)
#        out_features.append(A2s)


        for i in range(self.n_train_es):
            out_features.append(theano.tensor.tensordot(Es, self.w[weight_index,:,:], axes=[1,0]))
            weight_index += 1

        for i in range(self.n_train_ms):
            out_features.append(theano.tensor.tensordot(Ms, self.w[weight_index,:,:], axes=[1,0]))
            weight_index += 1

        for i in range(self.n_train_pts):
            out_features.append(theano.tensor.tensordot(Pts, self.w[weight_index,:,:], axes=[1,0]))
            weight_index += 1

        for i in range(self.n_train_a1s):
            out_features.append(theano.tensor.tensordot(A1s, self.w[weight_index,:,:], axes=[1,0]))
            weight_index += 1

        for i in range(self.n_train_a2s):
            out_features.append(theano.tensor.tensordot(A2s, self.w[weight_index,:,:], axes=[1,0]))
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
        magic_sum  = magic1-magic2


        # Build d_ij^2 = (k_i - k_j)^mu (k_i - k_j)^nu eta_mu_nu

        # b f p p'
        # x * magic  gives b f p^2, reshape to b f p p'
        d2_ij = K.reshape(K.expand_dims(K.dot(x, magic_diff), -1), (x.shape[0], x.shape[1], x.shape[2], x.shape[2]))
        
        # elements squared
        d2_ij = K.pow(d2_ij, 2)

        # fold with the metric
        # b f p p' * f  = b p p'  

        for i in range(self.n_train_dijs):

            if self.debug:
                print("metric:")
                print(K.eval(self.m[metric_index,:].shape))      
                print(K.eval(self.m[metric_index,:]))

            if self.debug:
                print("d2_ij:")
                print(K.eval(d2_ij.shape))
                print(K.eval(d2_ij))
            
            if self.train_metric:
                metric = self.m[metric_index,:]
                metric_index += 1
            else:
                metric = K.variable(np.array([ -1., 1., 1., 1.]))
                
            m_d2_ij = theano.tensor.tensordot(d2_ij, metric, axes=[1,0])

            out_features.append(K.sum(theano.tensor.tensordot(m_d2_ij, self.w[weight_index,:,:], axes=[1,0]), axis=2))
            weight_index += 1

        if self.debug:
            print("Done with d2_ijs")

        # Build d_ij^2 = (k_i - k_j)^mu (k_i - k_j)^nu eta_mu_nu

        # b f p p'
        # x * magic  gives b f p^2, reshape to b f p p'
        d2_3d_ij = K.reshape(K.expand_dims(K.dot(x, magic_diff), -1), (x.shape[0], x.shape[1], x.shape[2], x.shape[2]))
        
        # elements squared
        d2_3d_ij = K.pow(d2_3d_ij, 2)

        # fold with the metric
        # b f p p' * f  = b p p'  
        d2_3d_ij = theano.tensor.tensordot(d2_3d_ij, metric3, axes=[1,0])

        for i in range(self.n_train_dijs_3d):
            out_features.append(K.sum(theano.tensor.tensordot(d2_3d_ij, self.w[weight_index,:,:], axes=[1,0]), axis=2))
            weight_index += 1


        # Build m_ij^2 = (k_i + k_j)^mu (k_i + k_j)^nu eta_mu_nu

        # b f p p'
        # x * magic  gives b f p^2, reshape to b f p p'
        m2_ij = K.reshape(K.expand_dims(K.dot(x, magic_sum), -1), (x.shape[0], x.shape[1], x.shape[2], x.shape[2]))
        
        # elements squared
        m2_ij = K.pow(m2_ij, 2)

        # fold with the metric
        # b f p p' * f  = b p p'  

        for i in range(self.n_train_mijs):

            if self.train_metric:
                metric = self.m[metric_index,:]
                metric_index += 1
            else:
                metric = K.variable(np.array([ -1., 1., 1., 1.]))

            m_m2_ij = theano.tensor.tensordot(m2_ij, metric, axes=[1,0])

            out_features.append(K.min(theano.tensor.tensordot(m_m2_ij, self.w[weight_index,:,:], axes=[1,0]), axis=2))
            weight_index += 1

        if self.debug:
            print("done with m2_ij")

        # Build m_ij^2 = (k_i + k_j)^mu (k_i + k_j)^nu (only 3metric)

        # b f p p'
        # x * magic  gives b f p^2, reshape to b f p p'
        m2_3d_ij = K.reshape(K.expand_dims(K.dot(x, magic_sum), -1), (x.shape[0], x.shape[1], x.shape[2], x.shape[2]))
        
        # elements squared
        m2_3d_ij = K.pow(m2_3d_ij, 2)

        # fold with the metric
        # b f p p' * f  = b p p'  
        m2_3d_ij = theano.tensor.tensordot(m2_3d_ij, metric3, axes=[1,0])

        for i in range(self.n_train_mijs_3d):
            out_features.append(K.sum(theano.tensor.tensordot(m2_3d_ij, self.w[weight_index,:,:], axes=[1,0]), axis=2))
            weight_index += 1

 
        # Build cos_ij = m_ij^2 / 2 E_i E_j

        Ei = K.expand_dims(Es,-1) # bpN
        Ej = K.expand_dims(Es,-2) # bNp
        Eij = Ei*Ej # bpp
        Eij = K.clip(Eij, 0.0001, 100000000.)

        ratio_m2ij_Eij = m2_ij * pow(Eij,-1)

        onemat = K.ones((self.n_particles, self.n_particles))
        onemat = K.expand_dims(onemat, 0)

        cos_ij = onemat + ratio_m2ij_Eij

        for i in range(self.n_train_cosijs):
            out_features.append(K.sum(theano.tensor.tensordot(cos_ij, self.w[weight_index,:,:], axes=[1,0]), axis=2))
            weight_index += 1


#        if self.debug:
#            print ("Eij:")
#            print (K.eval(Eij))
#            print (K.eval(Eij.shape))
#            print ("sum_ij:")
#            print (K.eval(sum_ij))
#            print (K.eval(sum_ij.shape))
#            print ("ratio:")
#            print (K.eval(ratio_dij_Eij))
#            print (K.eval(ratio_dij_Eij.shape))
#            print ("cos:")
#            print (K.eval(cos))
#            print (K.eval(cos.shape))


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

