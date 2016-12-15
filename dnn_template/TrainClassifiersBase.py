########################################
# Imports
########################################

print("Imports: Starting...")

import socket
hostname = socket.gethostname()

print("We are at:", hostname)

import sys

def fixPath():
    newpath = []
    for v in sys.path:
        if "cvmfs" in v and "pandas" in v:
            continue
        newpath += [v]
    return newpath

sys.path = fixPath()

import os
import pickle
import pdb

print("Imported basics")

if "t3ui" in hostname:
    import ROOT
    print("Imported ROOT")

import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm

import pandas

if "t3ui" in hostname:
    import root_numpy
    from matplotlib.colors import LogNorm

import h5py

print("Imported numpy+friends")



from keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD
from keras.utils import np_utils, generic_utils
from keras.layers.advanced_activations import PReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, ZeroPadding3D
from keras.layers.core import Reshape
from keras.models import model_from_yaml

print("Imported keras")

if "t3ui" in hostname:
    import sklearn
    from sklearn import preprocessing
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.tree  import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.multiclass import OutputCodeClassifier
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.multiclass import OneVsOneClassifier
    from sklearn.preprocessing import normalize
    from sklearn.svm import LinearSVC
    from sklearn.preprocessing import StandardScaler  
    print("Imported sklearn")

from plotlib import *

print("Imports: Done...")


class LossPlotter(Callback):

    def __init__(self):
        super(Callback, self).__init__()

        self.loss_hist = []
        self.val_loss_hist = []
        

    def on_train_begin(self, logs={}):
        pass

    def on_batch_end(self, batch, logs={}):
        pass

    def on_epoch_end(self, epoch, logs):

        self.loss_hist.append(logs["loss"])
        self.val_loss_hist.append(logs["val_loss"])

        plt.clf()
        plt.plot(self.loss_hist)
        plt.plot(self.val_loss_hist)
        plt.savefig("loss_{0}.png".format(epoch))

        

########################################
# Class: Classifier
########################################

class Classifier:
    def __init__(self,
                 name,
                 backend,
                 params,
                 load_from_file,
                 datagen_train,
                 datagen_test,
                 model,
                 image_fun,
                 class_names,
                 inpath = ".",
                 plot_name = "",                 
                 varlist = [],
             ):
        self.name = name
        self.backend = backend
        self.params = params
        self.load_from_file = load_from_file
        self.datagen_train = datagen_train
        self.datagen_test  = datagen_test
        self.model = model
        self.image_fun = image_fun
        self.inpath = inpath
        
        self.class_names = class_names
        self.classes = sorted(class_names.keys())
        
        self.varlist = varlist

        if plot_name:
            self.plot_name = plot_name
        else:
            self.plot_name = name

    def prepare(self):

        if not self.load_from_file:
            if self.backend == "scikit":
                train_scikit(self)
            elif self.backend == "keras":
                train_keras(self)
        else:
            if self.backend == "scikit":
                f = open(os.path.join(self.inpath,self.name + ".pickle"), "r")
                self.model = pickle.load(f)
                f.close()
            elif self.backend == "keras":

                print("Loading", self.name)
                f = open(os.path.join(self.inpath,self.name + ".yaml"), "r")
                yaml_string = f.read()
                f.close()       
                print("Getting yaml")
                self.model = model_from_yaml(yaml_string)                
                print("Got yaml")
                self.model.load_weights(os.path.join(self.inpath,self.name + "_weights.h5"))
                        
                print("Loading", self.name, "from file: Done...")
                #print "Now training a bit"
                #train_keras(self)


########################################
# Helper: train_scitkit
########################################

def train_scikit(clf):

    print("Starting train_scikit with the parameters: ")
    for k,v in clf.params.items():
        print("\t", k,"=",v)
    print("Classifier name:",clf.name)
        
    df = next(clf.datagen_train)

    # Shuffle
    df = df.iloc[np.random.permutation(len(df))]

    X = get_data_vars(df, clf.varlist)
    y = df["is_signal_new"].values    
    
    clf.model.fit(X, y)

    f = open(clf.name + ".pickle","wb")
    pickle.dump(clf.model, f)
    f.close()


########################################
# Helper: train_keras
########################################

def train_keras(clf):

    print("Starting train_keras with the parameters: ")
    for k,v in clf.params.items():
        print("\t", k,"=",v)
      
    # Prepare model and train
    sgd = SGD(lr = clf.params["lr"], 
              decay = clf.params["decay"], 
              momentum = clf.params["momentum"], 
              nesterov=True)
    clf.model.compile(loss='mean_squared_error', optimizer=sgd, metrics=["accuracy"])
                
    print("Calling fit_generator")

    def generator(dg):
        while True:
            df = next(dg)
            
            # Shuffle
            df = df.iloc[np.random.permutation(len(df))]

            X = clf.image_fun(df)
            y = np_utils.to_categorical(df["is_signal_new"].values,2)

            yield X,y

        
    train_gen = generator(clf.datagen_train)
    test_gen  = generator(clf.datagen_test)

    early_stop = EarlyStopping(monitor='val_loss', 
                               patience=20, 
                               verbose=0, 
                               mode='auto')

    filepath="weights-" + clf.name + "-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='auto')


    ret = clf.model.fit_generator(train_gen,
                                  samples_per_epoch = clf.params["samples_per_epoch"],
                                  nb_epoch = clf.params["nb_epoch"],
                                  verbose=2, 
                                  validation_data=test_gen,
                                  nb_val_samples = clf.params["samples_per_epoch"],
                                  callbacks = [checkpoint, early_stop, LossPlotter()])

    print("Done")

    plt.clf()
    plt.plot(ret.history["acc"])
    plt.plot(ret.history["val_acc"])
    plt.savefig(clf.name + "_acc.png")

    plt.clf()
    plt.plot(ret.history["loss"])
    plt.plot(ret.history["val_loss"])
    plt.savefig(clf.name + "_loss.png")

    valacc_out = open(clf.name + "_valacc.txt", "w")
    valacc_out.write(str(ret.history["val_acc"][-1]) + "\n")
    valacc_out.close()

    maxvalacc_out = open(clf.name + "_maxvalacc.txt", "w")
    maxvalacc_out.write(str(max(ret.history["val_acc"])) + "\n")
    maxvalacc_out.close()
  
    deltaacc_out = open(clf.name + "_deltaacc.txt", "w")
    deltaacc_out.write(str(ret.history["val_acc"][-1] - ret.history["acc"][-1]) + "\n")
    deltaacc_out.close()

    # save the architecture
    model_out_yaml = open(clf.name + ".yaml", "w")
    model_out_yaml.write(clf.model.to_yaml())
    model_out_yaml.close()
  
    # And the weights
    clf.model.save_weights(clf.name + '_weights.h5', overwrite=True)

    

########################################
# Helper: rocplot
########################################

def rocplot(clf, df):
    
    print("rocplot", clf.name)

    nbins = 100
    min_prob = min(df["sigprob_"+ clf.name])
    max_prob = max(df["sigprob_"+ clf.name])
        
    if min_prob >= max_prob:
        max_prob = 1.1 * abs(min_prob)
        
    plt.clf()
                
    # Signal 
    h1 = make_df_hist((nbins*5,min_prob,max_prob), df.loc[df["is_signal_new"] == 1,"sigprob_"+clf.name])    
    
    # Background
    h2 = make_df_hist((nbins*5,min_prob,max_prob), df.loc[df["is_signal_new"] == 0,"sigprob_"+clf.name])    

    # And turn into ROC
    r, e = calc_roc(h1, h2)

    plt.clf()        
    plt.plot(r[:, 0], r[:, 1], lw=1, ls="--")
        
    # Setup nicely
    plt.legend(loc=2)
    plt.xlabel( "signal match efficiency", fontsize=16)
    plt.ylabel("fake match efficiency", fontsize=16)
    plt.legend(loc=2)

    plt.yscale('log')
    plt.xlim(0,1)
    plt.ylim(0.001,1)
    
    plt.show()

    plt.savefig(clf.name + "-ROC.png")


    plt.clf()        
    plt.plot(r[:, 0], 1./r[:, 1], lw=1, ls="--")
        
    # Setup nicely
    plt.legend(loc=2)
    plt.xlabel( "signal match efficiency", fontsize=16)
    plt.ylabel("1/fake match efficiency", fontsize=16)
    plt.legend(loc=2)

    plt.xlim(0,1)
    plt.ylim(1,100000)


    plt.yscale('log')    
    plt.show()

    plt.savefig(clf.name + "-ROC-inv.png")


########################################
# Helper: rocplot_multi
########################################

def rocplot_multi(classifiers, dfs, labels = [], styles = [],suffix =""):

    if not labels:
        labels = classifiers

    if not styles:
        styles = ["--"] * len(classifiers)
    
    rocs = []
    
    for clf_name,df in zip(classifiers, dfs):
        nbins = 100
        min_prob = min(df["sigprob_"+ clf_name])
        max_prob = max(df["sigprob_"+ clf_name])

        if min_prob >= max_prob:
            max_prob = 1.1 * abs(min_prob)

        # Signal 
        h1 = make_df_hist((nbins*5,min_prob,max_prob), df.loc[df["is_signal_new"] == 1,"sigprob_"+clf_name])    

        # Background
        h2 = make_df_hist((nbins*5,min_prob,max_prob), df.loc[df["is_signal_new"] == 0,"sigprob_"+clf_name])    

        # And turn into ROC
        r, e = calc_roc(h1, h2)
        
        rocs.append(r)
        
    plt.clf()        
    for clf_name, roc, label, ls in zip(classifiers, rocs, labels, styles):
        plt.plot(roc[:, 0], 
                 1./roc[:, 1], 
                 lw=1, 
                 label = label,
                 ls=ls)
                
    # Setup nicely
    plt.legend(loc=2)
    plt.xlabel( "signal match efficiency", fontsize=16)
    plt.ylabel("1/fake match efficiency", fontsize=16)
    plt.legend(loc=1)

    plt.xlim(0,2.0)
    plt.ylim(1,100000)

    plt.yscale('log')    
    plt.show()

    plt.savefig("multi-ROC-inv"+suffix+".png")



    plt.xlim(0.2,0.5)
    plt.ylim(30,80)

    plt.yscale('linear')    
    plt.show()

    plt.savefig("multi-ROC-inv-zoom"+suffix+".png")


########################################
# Helper: datagen
########################################

def datagen(sel, brs, infname_sig, infname_bkg, n_chunks=10):

    f_sig = ROOT.TFile.Open(infname_sig)
    sig_entries = f_sig.Get("tree").GetEntries()
    f_sig.Close()

    f_bkg = ROOT.TFile.Open(infname_bkg)
    bkg_entries = f_bkg.Get("tree").GetEntries()
    f_bkg.Close()

    # Initialize
    step_sig = sig_entries/n_chunks
    step_bkg = bkg_entries/n_chunks

    i_start_sig = 0
    i_start_bkg = 0        

    # Generate data forever
    while True:
        
        d_sig = root_numpy.root2array(infname_sig, treename="tree", branches=brs, selection = sel, start=i_start_sig, stop = i_start_sig + step_sig)
        d_bkg = root_numpy.root2array(infname_bkg, treename="tree", branches=brs, selection = sel, start=i_start_bkg, stop = i_start_bkg + step_bkg)

        

        i_start_sig += step_sig
        i_start_bkg += step_bkg
        # roll over
        if ((i_start_sig + step_sig >= sig_entries) or 
            (i_start_bkg + step_bkg >= bkg_entries)):
            i_start_sig = 0
            i_start_bkg = 0

        dfs = []
                
        for is_signal, d in enumerate([d_bkg, d_sig]):
            
            df = pandas.DataFrame(d['entry'],columns=["entry"])

            for br in brs:

                if br in ["entry","img", "img_dr", "img_e", "img_et", "img_min"]:
                    pass
                else:
                    df[br] = d[br]

            
            for i in range(1600):

                #df["img_{0}".format(i)] = d["img"][:,i]
                #df["img_dr_{0}".format(i)] = d["img_dr"][:,i]
                #df["e{0}".format(i)]  = d["img_e"][:,i]
                df["et{0}".format(i)] = d["img_min"][:,i]
            
            df["is_signal_new"] = is_signal

            dfs.append(df)


        df = pandas.concat(dfs, ignore_index=True)

        yield df

#        # We need to do a bit of numpy magic to properly convery the input
#        # an array with heterogenous dimensions (so a int + a list of ints)
#        # makes pandas unhappy. But if we cast the int and the list both as
#        # 'object' it works
#        datatype = [ (br,object) for br in brs ]
#        
#        # and we have to cast astype(object) in between for this to work..
#        df_sig = pandas.DataFrame(np.asarray(d_sig.astype(object),dtype=datatype))
#        df_sig["is_signal_new"] = 1
#
#        df_bkg = pandas.DataFrame(np.asarray(d_bkg.astype(object),dtype=datatype))
#        df_bkg["is_signal_new"] = 0
#
#        
#                    


########################################
# Helper: datagen_batch
########################################

def datagen_batch(sel, brs, infname_sig, infname_bkg, n_chunks=10, batch_size=1024):
    """Generates data in batches 
    This uses the datagen as input which reads larger chungs from the file at once
    One batch is what we pass to the classifiert for training at once, so we want this to be
    finer than the "chunksize" - these just need to fit in the memory. """

    print("Welcome to datagen_batch")

    # Init the generator that reads from the file    
    get_data = datagen(sel=sel, 
                       brs=brs, 
                       infname_sig=infname_sig, 
                       infname_bkg=infname_bkg, 
                       n_chunks=n_chunks)


    df = []    
    i_start = 0

    print("get_data finished")

    while True:

        if len(df)>i_start+batch_size:            
            yield df[i_start:i_start+batch_size]
            i_start += batch_size
        else:
            # refill data stores            
            df = next(get_data)
            i_start = 0



########################################
# Helper: datagen_batch
########################################

def datagen_batch_h5(brs, infname, batch_size=1024):
    """Generates data in batchaes using partial reading of h5 files """

    store = pandas.HDFStore(infname)
    size = store.get_storer('table').nrows    

    i_start = 0
    
    while True:

            
        if size > i_start+batch_size:            
            foo = store.select('table',
                               columns = brs,
                               start = i_start,
                               stop  = i_start + batch_size)
                        
            yield foo
            i_start += batch_size
        else:
            i_start = 0


def analyze(clf):

    if clf.backend == "keras":
        # Prepare the model
        sgd = SGD(lr = clf.params["lr"], 
                  decay = clf.params["decay"], 
                  momentum = clf.params["momentum"], 
                  nesterov=True)
        clf.model.compile(loss='mean_squared_error', optimizer=sgd, metrics=["accuracy"])

    # How many batches to process?
    # We should make one iteration on the whole file here
    

    nbatches = clf.params["samples_per_epoch"]/clf.params["batch_size"]

    df_all = pandas.DataFrame()
    
    # Loop over batches
    for i_batch in range(nbatches):

        print("At ", i_batch, "/", nbatches)

        df = next(clf.datagen_test)

        if clf.backend == "keras":
            X = clf.image_fun(df)        
            probs = clf.model.predict_on_batch(X)

            # predict_on_batch returns two values per image: 
            # signal and background probability
            # we're just interested in the signal prob (bg prob = 1 - signal_prob)        
            df["sigprob"] = probs[:,1] 

            # Now that we have calculated the classifier response, 
            # remove image from dataframe
            for i in range(1600):
                df.drop(["img_{0}".format(i)],axis=1)

        else:        
            X = get_data_vars(df, clf.varlist)
            probs = clf.model.predict_proba(X)
            
            df["sigprob"] = probs[:,1] 


        df_all = df_all.append(df)

    # Extra variables
    df_all["tau32_sd"] = df_all["tau3_sd"].astype(float)/df_all["tau2_sd"].astype(float)
    df_all["tau32"]    = df_all["tau3"].astype(float)/df_all["tau2"].astype(float)
                
    # Prepare ROC plot
    rocplot(clf, df_all)
        
    plots = []

    # Plot DNN Output 
    plots.append(["sigprob", [], 100,0,1, "sigprob"])

    # Other properties of top
    # Plot inclusive as well as in slices of DNN Output
    proto_plots = [["softdropped.M()", 100,0,300, "mass_sd"],
                   ["filtered.M()", 100,0,300, "mass_filt"],
                   ["fatjet.M()", 100,0,300, "mass_ungroomed"],
                   ["softdropped.Pt()", 100,0,500, "pt_sd"],
                   ["filtered.Pt()", 100,0,500, "pt_filt"],
                   ["fatjet.Pt()", 100,300,500, "pt_ungroomed"],
                   ["tau32_sd", 100,0,1, "tau32_sd"],
                   ["tau32", 100,0,1, "tau32"]]                                      
    for [variable, nbins, xmin, xmax, name] in proto_plots:
        plots.append([variable, [], nbins, xmin, xmax, name])
        plots.append([variable, [df_all["sigprob"] > 0.8], nbins, xmin, xmax, name + "_hi"])
        plots.append([variable, [df_all["sigprob"] > 0.4, df_all["sigprob"] < 0.6], nbins, xmin, xmax, name + "_md"])
        plots.append([variable, [df_all["sigprob"] < 0.2], nbins, xmin, xmax, name + "_lo"])


    # Make all plots
    for plot in plots:
                         
        [variable, cuts, nbins, xmin, xmax, name] = plot
        
        cut_sig = reduce(lambda x,y:x&y,cuts + [(df_all["is_signal_new"] == 1)])
        cut_bkg = reduce(lambda x,y:x&y,cuts + [(df_all["is_signal_new"] == 0)])

        sig = df_all.loc[cut_sig,variable]
        bkg = df_all.loc[cut_bkg,variable]
            
        plt.clf()
        plt.hist(sig, label="sig", bins=np.linspace(xmin,xmax,nbins), alpha=0.4, normed=True)
        plt.hist(bkg, label="bkg", bins=np.linspace(xmin,xmax,nbins), alpha=0.4, normed=True)
        plt.xlabel(variable, fontsize=16)
        plt.ylabel("Fraction of jets", fontsize=16)        
        plt.legend(loc=1)
        plt.xlim(xmin,xmax)
        plt.show()
        plt.savefig(clf.name + name)


    # And 2D Plots:
    prob_sig = df_all.loc[(df_all["is_signal_new"] == 1),"sigprob"]
    prob_bkg = df_all.loc[(df_all["is_signal_new"] == 0),"sigprob"]
    for var in ["softdropped.M()" ,"filtered.M()", "fatjet.M()", 
                "softdropped.Pt()","filtered.Pt()", "fatjet.Pt()", 
                "tau32_sd", "tau32"]:

        var_sig = df_all.loc[(df_all["is_signal_new"] == 1), var]
        var_bkg = df_all.loc[(df_all["is_signal_new"] == 0), var]

        name = var.replace("(","").replace(")","").replace(".","_")
        
        plt.clf()
        plt.hexbin(var_sig, prob_sig)
        plt.show()   
        plt.savefig(clf.name + "-2d-" + name + "-sig.png")

        plt.clf()
        plt.hexbin(var_bkg, prob_bkg)
        plt.show()   
        plt.savefig(clf.bane + "-2d-" + name + "-bkg.png")




def analyze_multi(classifiers):

    # Analyzes a list of classifiers to study correlations, etc.
    # Use the first classifier for settings/datagen/etc

    # Prepare all neural networks
    for clf in classifiers:
        if clf.backend == "keras":
            
            # Prepare the model
            sgd = SGD(lr = clf.params["lr"], 
                      decay = clf.params["decay"], 
                      momentum = clf.params["momentum"], 
                      nesterov=True)
            clf.model.compile(loss='mean_squared_error', optimizer=sgd, metrics=["accuracy"])

    
    nbatches = classifiers[0].params["samples_per_epoch"]/classifiers[0].params["batch_size"]

    df_all = pandas.DataFrame()
    
    # Loop over batches
    for i_batch in range(nbatches):

        print("At ", i_batch, "/", nbatches)

        df = next(classifiers[0].datagen_test)

        for clf in classifiers:

            if clf.backend == "keras":
                X = clf.image_fun(df)        
                probs = clf.model.predict_on_batch(X)
            else:        
                X = get_data_vars(df, clf.varlist)
                probs = clf.model.predict_proba(X)
 
            # prediction returns two values: 
            # signal and background probability
            # we're just interested in the signal prob (bg prob = 1 - signal_prob)        
            df["sigprob_" + clf.name] = probs[:,1] 

        # Now that we have calculated the classifier response, 
        # remove image from dataframe
        for i in range(1600):
            df.drop(["img_{0}".format(i)],axis=1)

        df_all = df_all.append(df)

    store_df = pandas.HDFStore('test-store.h5')
    store_df["all"] = df_all

    # Individual ROCs
    for clf in classifiers:
        rocplot(clf, df_all)
        



#    # Calculate extra variables
#    df_all["tau32_sd"] = df_all["tau3_sd"].astype(float)/df_all["tau2_sd"].astype(float)
#    df_all["tau32"]    = df_all["tau3"].astype(float)/df_all["tau2"].astype(float)
#                
#    # Make ROC plots
#    # Together
#    rocplot_multi(classifiers, df_all)
#
#
#    plots = []
#
#    # Plot Classifier outputs
#    for clf in classifiers:
#        plots.append(["sigprob_"+clf.name, [], 100,0,1, "sigprob_"+clf.name])
#
#    # Other properties of top
#    # Plot inclusive as well as in slices of DNN Output
#    proto_plots = [["softdropped.M()", 100,0,300, "mass_sd"],
#                   ["filtered.M()", 100,0,300, "mass_filt"],
#                   ["fatjet.M()", 100,0,300, "mass_ungroomed"],
#                   ["softdropped.Pt()", 100,0,500, "pt_sd"],
#                   ["filtered.Pt()", 100,0,500, "pt_filt"],
#                   ["fatjet.Pt()", 100,300,500, "pt_ungroomed"],
#                   ["tau32_sd", 100,0,1, "tau32_sd"],
#                   ["tau32", 100,0,1, "tau32"]]                                      
#    for [variable, nbins, xmin, xmax, name] in proto_plots:
#        plots.append([variable, [], nbins, xmin, xmax, name])
#        #plots.append([variable, [df_all["sigprob"] > 0.8], nbins, xmin, xmax, name + "_hi"])
#        #plots.append([variable, [df_all["sigprob"] > 0.4, df_all["sigprob"] < 0.6], nbins, xmin, xmax, name + "_md"])
#        #plots.append([variable, [df_all["sigprob"] < 0.2], nbins, xmin, xmax, name + "_lo"])
#
#
#    # Make all plots
#    for plot in plots:
#                         
#        [variable, cuts, nbins, xmin, xmax, name] = plot
#        
#        cut_sig = reduce(lambda x,y:x&y,cuts + [(df_all["is_signal_new"] == 1)])
#        cut_bkg = reduce(lambda x,y:x&y,cuts + [(df_all["is_signal_new"] == 0)])
#
#        sig = df_all.loc[cut_sig,variable]
#        bkg = df_all.loc[cut_bkg,variable]
#            
#        plt.clf()
#        plt.hist(sig, label="sig", bins=np.linspace(xmin,xmax,nbins), alpha=0.4, normed=True)
#        plt.hist(bkg, label="bkg", bins=np.linspace(xmin,xmax,nbins), alpha=0.4, normed=True)
#        plt.xlabel(variable, fontsize=16)
#        plt.ylabel("Fraction of jets", fontsize=16)        
#        plt.legend(loc=1)
#        plt.xlim(xmin,xmax)
#        plt.show()
#        plt.savefig(name)
#
#
##    # And 2D Plots:
##    prob_sig = df_all.loc[(df_all["is_signal_new"] == 1),"sigprob"]
##    prob_bkg = df_all.loc[(df_all["is_signal_new"] == 0),"sigprob"]
##    for var in ["softdropped.M()" ,"filtered.M()", "fatjet.M()", 
##                "softdropped.Pt()","filtered.Pt()", "fatjet.Pt()", 
##                "tau32_sd", "tau32"]:
##
##        var_sig = df_all.loc[(df_all["is_signal_new"] == 1), var]
##        var_bkg = df_all.loc[(df_all["is_signal_new"] == 0), var]
##
##        name = var.replace("(","").replace(")","").replace(".","_")
##        
##        plt.clf()
##        plt.hexbin(var_sig, prob_sig)
##        plt.show()   
##        plt.savefig(clf.name + "-2d-" + name + "-sig.png")
##
##        plt.clf()
##        plt.hexbin(var_bkg, prob_bkg)
##        plt.show()   
##        plt.savefig(clf.bane + "-2d-" + name + "-bkg.png")






def eval_single(clf, suffix=""):

    # Prepare all neural networks

    if clf.backend == "keras":

        # Prepare the model
        sgd = SGD(lr = clf.params["lr"], 
                  decay = clf.params["decay"], 
                  momentum = clf.params["momentum"], 
                  nesterov=True)
        clf.model.compile(loss='mean_squared_error', optimizer=sgd, metrics=["accuracy"])
    
    nbatches = int(clf.params["samples_per_epoch_test"]/clf.params["batch_size"] - 1)

    df_all = pandas.DataFrame()
    
    # Loop over batches
    for i_batch in range(nbatches):

        print("At ", i_batch, "/", nbatches)

        df = next(clf.datagen_test)

        if clf.backend == "keras":
            X = clf.image_fun(df)        
            probs = clf.model.predict_on_batch(X)
        else:        
            X = get_data_vars(df, clf.varlist)
            probs = clf.model.predict_proba(X)
 
        # prediction returns two values: 
        # signal and background probability
        # we're just interested in the signal prob (bg prob = 1 - signal_prob)        
        df["sigprob_" + clf.name] = probs[:,1] 

        # Now that we have calculated the classifier response, 
        # remove the rest
        cols_to_keep = set(["entry", "is_signal_new", "sigprob_" + clf.name])
        cols_to_drop = list(set(df.columns) - cols_to_keep)
        df = df.drop(cols_to_drop,axis=1)

        df_all = df_all.append(df)

    store_df = pandas.HDFStore('output_' + clf.name + suffix + '.h5')
    store_df["all"] = df_all


########################################
# Data access helpers
########################################

def get_data_vars(df, varlist):        
    return df[varlist].values

def get_data_flatten(df, varlist):
    
    # tmp is a 1d-array of 1d-arrays
    # so we need to convert it to 2d array
    tmp = df[varlist].values.flatten() # 
    ret = np.vstack([tmp[i] for i in xrange(len(tmp))])
     
    return ret 
