import pandas

from sklearn.metrics import confusion_matrix, classification_report, log_loss


import matplotlib as mpl
mpl.use('Agg')
import numpy as np
np.set_printoptions(precision=2)

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm

import itertools

import pdb

name = "deeph_test_6"
n_classes = 5


store = pandas.HDFStore("output_{0}.h5".format(name))

cnt = 0

def response(row):
    
    global cnt

    scores = [row[n] for n in ["cprob_{0}_{1}".format(i_class, name) for i_class in range(n_classes)]]
    
    if cnt%1000 == 0:
        print cnt
    
    cnt+=1
    
    for i_class in range(n_classes):
        if scores[i_class] == max(scores):
            return i_class
    
    return -1

df = store.select("all", stop=100)


df['label'] = df.apply(lambda row: response(row),axis=1)

conf_m =  confusion_matrix(df["class_new"], df["label"])


pdb.set_trace()


ll = log_loss(df["class_new"], df["label"],labels=[0,1,2,3,4])
print "Log loss: {0}".format(ll)

print classification_report(df["class_new"], df["label"])

np.set_printoptions(precision=2)
    
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)


    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{0:.3f}".format(cm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(conf_m, classes=["0","1","2","3","4"], normalize=True,
                      title='Normalized confusion matrix')
plt.savefig("norm.png")


