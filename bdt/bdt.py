import ROOT
import pandas
import root_numpy
import numpy as np
import matplotlib.pyplot as plt
import sys, optparse
import math

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals import joblib

#load data
op = optparse.OptionParser(usage=__doc__)
op.add_option("--treeplot", dest="TREEPLOT", default=False, action="store_true", help="Make a decision tree classifier plot")
op.add_option("--bounds", dest="BOUNDS", default=False, action="store_true", help="Plot decision tree boundaries in two input variables")

opts, args = op.parse_args()

infname_sig, infname_bkg = args[0], args[1]

#set up DataFrames
df_sig = pandas.DataFrame(root_numpy.root2rec(infname_sig, branches=["tau2_sd","tau3_sd","softdropjet.M()","fatjet.M()"]))
df_bkg = pandas.DataFrame(root_numpy.root2rec(infname_bkg, branches=["tau2_sd","tau3_sd","softdropjet.M()","fatjet.M()"]))

df_sig["is_signal"] = 1
df_bkg["is_signal"] = 0

df = pandas.concat([df_sig, df_bkg], ignore_index=True)
df = df.iloc[np.random.permutation(len(df))]

df_train = df[0:150000]
df_test_orig = df[150000:]

df_sig = np.asarray(df_sig)
df_bkg = np.asarray(df_bkg)
df_train = np.asarray(df_train)
df_test = np.asarray(df_test_orig)

# take tau2, tau3 and M
X_train, y_train = df_train[:,(0,1,2)], df_train[:,4]
X_test, y_test = df_test[:,(0,1,2)], df_test[:,4]

#hack: set nans to -0.1, this shouldn't happen anymore
for (ix, iy), val in np.ndenumerate(X_train):
    if math.isnan(val):
        print ix, iy
        X_train[ix][iy] = -0.1

# Create and fit an AdaBoosted decision tree
clf = DecisionTreeClassifier(max_depth=20)
bdt = AdaBoostClassifier(clf,
                         algorithm="SAMME",
                         n_estimators=10,
                         learning_rate=0.1)

bdt.fit(X_train,y_train)
clf.fit(X_train,y_train)

#joblib.dump(bdt,'bdt.pkl')
#sys.exit()
#bdt = joblib.load('bdt.pkl') 

# calculate the decision scores
twoclass_output = bdt.decision_function(X_train)
all_probs = bdt.predict_proba(X_test)
print type(all_probs)
dftt = df_test_orig.copy()
class_names = {0: "background",
               1: "signal"}
classes = sorted(class_names.keys())
for cls in classes:
    print type(cls)
    dftt[class_names[cls]] = all_probs[:,cls]
sig = dftt["is_signal"]==1
bkg = dftt["is_signal"]==0

probs = dftt["signal"][sig].values
probb = dftt["signal"][bkg].values

es, eb  = [], []

for c in np.arange(-1,1,0.01):
    es.append((float((probs > c).sum())/probs.size))
    eb.append((float((probb > c).sum())/probb.size))

print np.c_[es, eb]
np.savetxt('roc_test.dat',np.c_[es,eb])

# make DecisionTree classifier plot, requires sklearn v>=0.17
if(opts.TREEPLOT):

    from sklearn.externals.six import StringIO
    from sklearn.tree import export_graphviz
    import pydot

    features = ["Jet Mass", "tau3","tau2"]
    classes = ["background","signal"]
    dot_data = StringIO() 
    export_graphviz(clf, out_file=dot_data,
                    feature_names=features,  
                    class_names=classes,  
                    filled=True, rounded=True,  
                    special_characters=True)
    graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
    graph.write_pdf("classifier_tree.pdf")

#Plot the decision boundaries, only works for 2 features
if(opts.BOUNDS):

    plot_colors = "br"
    plot_step = 0.02
    class_names = "AB"

    plt.figure(figsize=(10, 5))

    # Plot the decision boundaries
    plt.subplot(121)
    x_min, x_max = X_train[:, 0].min(), X_train[:, 0].max()
    y_min, y_max = X_train[:, 1].min(), X_train[:, 1].max()
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))

    Z = bdt.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
    plt.axis("tight")

    labels = ["background","signal"]

    # Plot the training points
    for i, n, c in zip(range(2), class_names, plot_colors):
        idx = np.where(y_train == i)
        plt.scatter(X_train[idx, 0], X_train[idx, 1],
                c=c, cmap=plt.cm.Paired,
                label=labels[i],s=5.)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.legend(loc='upper right')
    plt.ylabel('Jet Mass') # or whatever it is
    plt.xlabel(r'$\tau_{3}$')
    plt.title('Decision Boundary')

    # Plot the two-class decision scores
    plot_range = (twoclass_output.min(), twoclass_output.max())
    plt.subplot(122)
    for i, n, c in zip(range(2), class_names, plot_colors):
        plt.hist(twoclass_output[y_train == i],
                 bins=100,
                 range=plot_range,
                 facecolor=c,
                 label=labels[i],
                 alpha=.5)
    x1, x2, y1, y2 = plt.axis()
    plt.axis((x1, x2, y1, y2 * 1.2))
    plt.legend(loc='upper right')
    plt.ylabel('Samples')
    plt.xlabel('Score')
    plt.title('Decision Scores')
    plt.subplots_adjust(wspace=0.35)
    plt.savefig('decision_bounds+scores.pdf')


sys.exit()

# plot roc-curve    
plt.subplot(121)    
print np.c_[es,eb]
plt.scatter(es,eb)
#plt.show()

