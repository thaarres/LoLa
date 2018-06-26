import sys,os
import pandas
import tensorflow
import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import np_utils, plot_model, vis_utils
import matplotlib as mpl
import matplotlib.pyplot as plt

import time 
tic = time.time()

print("Done importing Keras models and layers")

from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
print("Done importing SK Learn")

NC = 20

params = {"input_path"              : "/data/taarre/",
          "output_path"             : "./",
          "inputs"                  : "constit_lola",
          "model_name"              : "wLola_v1",        
          "nb_epoch"                : 1,
          "batch_size"              : 1024,
          "n_constit"               : NC,
          "n_features"              : 4,
          "name_train"              : "Wconst-train-v2_20nc_datacols-resort.h5",
          "name_test"               : "Wconst-test-v2_20nc_datacols-resort.h5",
          "name_val"                : "Wconst-val-v2_20nc_datacols-resort.h5",
          "read_from_file"          : False,
          "n_classes"               : 2,
          "signal_branch"           : "is_signal_new",
          "samples_per_epoch"       : None, # later filled from input files
          "add_weight"              : False, # If you want to use sample pt-weight
		  "weight_branch"           : "ptWeight",
		  
}		
		
def to_constit(df, n_constit, n_features):
	
	brs = []
	feat_list =  ["E","PX","PY","PZ"]  
	brs += ["{0}_{1}".format(feature,constit) for feature in feat_list for constit in range(n_constit)]
	ret = np.expand_dims(df[brs],axis=-1).reshape(-1, n_features, n_constit)  #expanded matrix: (1024, 4, 20)-->batch size, n_features, n_constituents
	ret = ret/500. #normalize values to be between -1 and 1 roughly
	return ret
	
def model_lola(params):

    model = Sequential() #linear stack of layers

    model.add(CoLa(input_shape = (params["n_features"], params["n_constit"]), #first layer, need to know input shape 4x20. 
                   add_total = True,
                   add_eye   = True,
                   n_out_particles = 15))

    model.add(LoLa(
        train_metric = False,
        es  = 0,
        xs  = 0,
        ys  = 0,
        zs  = 0,                 
        ms  = 1,                 
        pts = 1,                 
        n_train_es  = 1,
        n_train_ms  = 0,
        n_train_pts = 0,        
        n_train_sum_dijs   = 2,
        n_train_min_dijs   = 2))

    model.add(Flatten()) #From (4,20) to (None, 80)

    model.add(Dense(100))
    model.add(Activation('relu'))

    model.add(Dense(50))
    model.add(Activation('relu'))

    model.add(Dense(params["n_classes"], activation='softmax'))

    return model
	   
def getDF(store,truth_label,i_start,batch_size,ptmin,ptmax):
    foo = store.select('table',
                       columns = truth_label,
                       start = i_start,
                       stop  = i_start + batch_size,#),
                       where = 'jpt >= %s and jpt <= %s '%(ptmin,ptmax))
    return foo
    
def datagen_batch_h5(truth_label, infname, batch_size=1024):
    ptmin = 200
    ptmax = 5000
    """Generates data in batches using partial reading of h5 files """

    verbose = False

    if verbose:
        print("Opening " + infname)
    store = pandas.HDFStore(infname)
    # store.append('table', df, data_columns=df.head(1).reset_index().columns)
    size = store.get_storer('table').nrows   
    #print store.get_storer('table').group.table

    if verbose:
        print("Opened " + infname)

    i_start = 0
    step = 0
    
    while True:
       
        if size >= i_start+batch_size:  
            # print truth_label
            unvalid = True
            while unvalid:
                foo = getDF(store,truth_label,i_start,batch_size,ptmin,ptmax)
                i_start += batch_size
                step += 1
                unvalid = foo.empty
                        
            yield foo
            i_start += batch_size
            step += 1

            if size < i_start+batch_size:
                if verbose:
                    print("Closing " + infname)
                store.close()
                if verbose:
                    print("Closed " + infname)

        else:
            if verbose:
                print("Opening " + infname)
            store = pandas.HDFStore(infname)
            size = store.get_storer('table').nrows    
            if verbose:
                print("Opened " + infname)

            i_start = 0
						  
print("Parameters are:")
for k,v in params.items():
	print("{0}={1}".format(k,v))
truth_label = [params["signal_branch"]]
weight_label = None #If events are unweighted, set to none
if params["add_weight"]: 
	truth_label += [params["weight_branch"]] #make sure pTWeight gets picked up when generating samples
	weight_label = params["weight_branch"]
truth_label += ["jpt"]
pt_label = "jpt"
additional_variables = ["jsd0","jsd2","jN2sd0","jN2sd2","jtau21","jtau21sd0","jtau21sd2","jpt","jeta","jmass"]
feat_list =  ["E","PX","PY","PZ"] 
input_vector = ["{0}_{1}".format(feature,constit) for feature in feat_list for constit in range(params["n_constit"])]
print "input_vector = " ,input_vector

    # Reading H5FS
infname_train = os.path.join(params["input_path"], params["name_train"])
infname_test  = os.path.join(params["input_path"], params["name_test"])
infname_val   = os.path.join(params["input_path"], params["name_val"])

# H5FS: Count  training samples and define samples per epoch
store_train = pandas.HDFStore(infname_train)
n_train_samples = int((store_train.get_storer('table').nrows/params["batch_size"]))*params["batch_size"]

store_test = pandas.HDFStore(infname_test)
n_test_samples = int((store_test.get_storer('table').nrows/params["batch_size"]))*params["batch_size"]

store_val = pandas.HDFStore(infname_val)
n_val_samples = int((store_val.get_storer('table').nrows/params["batch_size"]))*params["batch_size"]

print("Total number of training samples = ", n_train_samples)
print("Total number of testing samples = ", n_test_samples)
print("Total number of valing samples = ", n_val_samples)
params["samples_per_epoch"] = n_train_samples
params["samples_per_epoch_test"] = n_test_samples
params["samples_per_epoch_val"] = n_val_samples


datagen_train = datagen_batch_h5(truth_label+input_vector, infname_train, batch_size=params["batch_size"]) 
datagen_test  = datagen_batch_h5(truth_label+input_vector, infname_test, batch_size=params["batch_size"])
datagen_val   = datagen_batch_h5(truth_label+additional_variables+input_vector, infname_val, batch_size=params["batch_size"])
#[1024 rows x 81 columns]: batch of 1024 training samples, with input vector 1+4*20 (label+feautures*constutuents)

sys.path.append("../LorentzLayer")
from cola import CoLa
from lola import LoLa

model     = model_lola(params)
image_fun = lambda x: to_constit(x,params["n_constit"], params["n_features"])

class_names = {}
for i in range(params["n_classes"]):
    class_names[i] = "c{0}".format(i)

classes = sorted(class_names.keys())
print("Starting train_keras ")
outdir = params["output_path"] + params["model_name"]

if not os.path.exists(outdir):
    os.makedirs(outdir)

model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=["accuracy"]) #TODO! Tried to add weighted_metrics=["accuracy"] but only works in Tensorflow
#Note: when using  categorical_crossentropy loss, targets should be in categorical format (e.g. if have 10 classes, target for each sample should be 10-dim vector that is all-0 expect for a 1 at index corresponding to class of the sample).
## import pydot; plot_model(model, to_file= 'model.eps',show_shapes=True,show_layer_names=True)

print("Calling fit_generator")

def generator(dg,params=params):
    while True:
        df = next(dg)

        # Shuffle
        df = df.iloc[np.random.permutation(len(df))]

        X = image_fun(df) #X matrix corresponding to 4 feauters, 20 particles and m training samples
        y = np_utils.to_categorical(df[params["signal_branch"]].values,params["n_classes"]) #convert integer targets into categorical targets
        if params["add_weight"]:
			w = df[params["weight_branch"]]
			yield X,y,w
        else:
			yield X,y
			

train_gen = generator(datagen_train) #return input, target. Continuously running until data is finished
val_gen  = generator(datagen_val)  #return input, target

early_stop = EarlyStopping(monitor='val_loss',
                           patience=5,
                           verbose=1,
                           mode='auto')

filepath= outdir + "/weights-latest.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto') #Save the model after every epoch. Only the one with smalles loss

# save the architecture
model_out_yaml = open(outdir + "/" + params["model_name"] + ".yaml", "w")
model_out_yaml.write(model.to_yaml())
model_out_yaml.close()

print("Steps: Train: {0} Validation: {1}".format(int(params["samples_per_epoch"]/params["batch_size"]),
                                           int(params["samples_per_epoch_val"]/params["batch_size"])))

ret = model.fit_generator(train_gen,#Fits model on data generated batch-by-batch by Python generator. The generator is run in parallel to  model, for efficiency.
                          steps_per_epoch = params["samples_per_epoch"]/params["batch_size"], #epoch finishes when this has been seen by model.Number of unique samples of dataset divided by batch size.
                          validation_steps = params["samples_per_epoch_val"]/params["batch_size"],
                          verbose=2,
                          epochs = params["nb_epoch"],
                          validation_data=val_gen,
                          # callbacks = [checkpoint, early_stop, LossPlotter(outdir)])
						  callbacks = [checkpoint, early_stop])

print("fit Done")
# list all data in history
print(ret.history.keys())



plt.clf()
plt.plot(ret.history["acc"])
plt.plot(ret.history["val_acc"])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(outdir + "/acc.png")

plt.clf()
plt.plot(ret.history["loss"])
plt.plot(ret.history["val_loss"])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(outdir + "/loss.png")


# save the architecture ...
# ...to yaml
model_out_yaml = open(outdir + "/" + params["model_name"] + ".yaml", "w")
model_out_yaml.write(model.to_yaml())
model_out_yaml.close()

#...to h5
model.save(outdir + "/" + params["model_name"] + '.h5') 

#...to json
model_out_json = model.to_json()
with open(outdir + "/" + params["model_name"] + ".json", "w") as json_file: json_file.write(model_out_json)

# And the weights
model.save_weights(outdir + "/" + params["model_name"] + '_weights.h5', overwrite=True)

# FINAL ROC
nbatches = int(params["samples_per_epoch_test"]/params["batch_size"] - 1)

# for layer in model.layers:
#     weights = layer.get_weights()
#     print(weights)


df_all = pandas.DataFrame()

# Loop over batches
for i_batch in range(nbatches):
    
    df = next(datagen_test)
    
    X = image_fun(df)        
    probs = model.predict_on_batch(X)
    
    
    # prediction returns two values: 
    # signal and background probability
    # we're just interested in the signal prob (bg prob = 1 - signal_prob)        
    df["sigprob_" + params["model_name"] ] = probs[:,1] 
    # df["jsd"] =
    # Label is maximum classiier
    df["label_" + params["model_name"] ]   = np.apply_along_axis(lambda x:int(np.argmax(x)),1,probs)
            
    # Add per-class probs
    probnames= []
    for iclass in range(params["n_classes"]):
        probname = "cprob_{0}_{1}".format(iclass, params["model_name"])            
        df[probname] = probs[:,iclass]
        probnames.append(probname)
  
    # Now that we have calculated the classifier response, 
    # remove the rest
	
	
    cols_to_keep = set(additional_variables+["entry", 
                        params["signal_branch"], 
                        "label_" + params["model_name"],
                        "sigprob_" + params["model_name"]] + probnames )
    cols_to_drop = list(set(df.columns) - cols_to_keep)
    df = df.drop(cols_to_drop,axis=1)
    
    df_all = df_all.append(df)
    
#pdb.set_trace()


cm = confusion_matrix(df_all[params["signal_branch"]], df_all["label_" + params["model_name"] ])
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print("Confusion Matrix:")
print(cm)

AOC = roc_auc_score(df_all[params["signal_branch"]], df_all["sigprob_" + params["model_name"] ])
print("AOC: {0}".format(AOC))

fpr, tpr, _ = roc_curve(df_all[params["signal_branch"]], df_all["sigprob_" + params["model_name"] ])

outdir = params["output_path"] + params["model_name"] 
plt.clf()
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {0:.2f})'.format(AOC))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.savefig(outdir + "/roc.png")

store_df = pandas.HDFStore('output_' + params["model_name"] + '.h5')
store_df["all"] = df_all

toc = time.time()
print 'TOTAL TRAINING TIME (min) == ', (toc-tic)/60
