import pdb

import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten

import pandas

from lola import *


verbose = False

# Generate dummy data
#np.random.seed(1)
data = np.random.random((10000, 4))

df = pandas.DataFrame(data)
df['signal']  = np.where(df[3] < 0.5, 1,0)

r_signal = 0.3
r_background = 0.1

def point_x(row):
    
    angle = row[2] * 2 * np.pi

    if row["signal"]:
        return row[0] + r_signal * np.cos(angle)
    else:
        return row[0] + r_background * np.cos(angle)

def point_y(row):
    
    angle = row[2] * 2 * np.pi

    if row["signal"]:
        return row[1] + r_signal * np.sin(angle)
    else:
        return row[1] + r_background * np.sin(angle)
    

df['x2'] = df.apply(lambda row:point_x(row),axis=1)
df['y2'] = df.apply(lambda row:point_y(row),axis=1)

data =  df[ [0,1,'x2','y2']].as_matrix()
data =  data.reshape((10000,2,2),order="F")


def test(
        dense = False,
        train_poly = False, 
        train_offset = False,
        train_metric = False ,
        debug = False,
):

    print "Start preapring model"

    model = Sequential()

    if dense:
        model.add(Dense(2, input_shape=(2, 2)))
        model.add(Activation('relu'))

    else:
        model.add(LoLa(input_shape=(2, 2),
                          train_poly = train_poly,
                          train_offset = train_offset,
                          train_metric = train_metric,
                          debug = debug,
                      ))

        #odel.add(LoLa(train_poly = train_poly,
        #                 train_offset = train_offset,
        #                 train_metric = train_metric,
        #                 debug = debug,
        #             ))


 
    model.add(Flatten())


    model.add(Dense(8))
    model.add(Activation('relu'))

    model.add(Dense(1, activation='sigmoid'))

    print "Done preparing model"

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    print "Train:"
    
    # Train the model, iterating on the data in batches of 32 samples
    ret = model.fit(data, df["signal"].as_matrix(), epochs=10, batch_size=32)

    if verbose:
        for layer in model.layers:
            weights = layer.get_weights()
            print weights


    return ret.history["loss"][-1]


fout = open("out.txt","w")

#for pattern in [
#        #[1,0,0,0],        
#        #[0,0,0,1],
#        #[0,0,1,0],
#        #[0,0,1,1],
#        #[0,1,0,0],
#        #[0,1,0,1],
#        [0,1,1,0],
#        #[0,1,1,1],
#]:
#                
#    n_iter = 1
#    losses = [test(*pattern) for _ in range(n_iter)]
#    
#    #fout.write("{0}: {1} {2}\n".format( pattern, np.mean(losses), np.median(losses)))

test(train_poly = True, train_offset = True, train_metric = False, debug = False)
