import ROOT
import pandas
import root_numpy
import numpy as np
import matplotlib.pyplot as plt
import sys, optparse
import math

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

#load data
op = optparse.OptionParser(usage=__doc__)
opts, args = op.parse_args()

infname_sig, infname_bkg = args[0], args[1]

#set up DataFrames
df_sig = pandas.DataFrame(root_numpy.root2rec(infname_sig, branches=["tau2_sd","tau3_sd","softdropjet.M()"]))
df_bkg = pandas.DataFrame(root_numpy.root2rec(infname_bkg, branches=["tau2_sd","tau3_sd","softdropjet.M()"]))

df_sig["tau3/tau2"] = df_sig["tau3_sd"]/df_sig["tau2_sd"]
df_bkg["tau3/tau2"] = df_bkg["tau3_sd"]/df_bkg["tau2_sd"]

df_sig["is_signal"] = 1
df_bkg["is_signal"] = 0

# some debugging
df_test = pandas.DataFrame(root_numpy.root2rec(infname_sig, branches=["tau2","tau3","fatjet.Pt()","fatjet.M()","tau2_sd","tau3_sd","softdropjet.Pt()","softdropjet.M()","softdropjet.E()","softdropjet.Px()","softdropjet.Py()","softdropjet.Pz()","tau2_filt","tau3_filt","filtered.Pt()","filtered.M()","htt_tagged"]))

#print df_test.loc[152]
#nans=0
#for ix, x in enumerate(np.asarray(df_test["tau3_sd"])):
#    if math.isnan(x):
#        print ix
#        nans+=1

df_sig = np.asarray(df_sig)
df_bkg = np.asarray(df_bkg)

#np.savetxt('sig.dat',df_sig)
#np.savetxt('bkg.dat',df_bkg)

X = np.concatenate((df_sig[:,(2,3)],df_bkg[:,(2,3)]))
y = np.concatenate((df_sig[:,4],df_bkg[:,4]))

#hack: set nans to -0.1
for (ix, iy), val in np.ndenumerate(X):
    if math.isnan(val):
#        print ix, iy
        X[ix][iy] = -0.1


# Create and fit an AdaBoosted decision tree
clf = DecisionTreeClassifier(max_depth=2)
bdt = AdaBoostClassifier(clf,
                         algorithm="SAMME",
                         n_estimators=20)

bdt.fit(X, y)
clf.fit(X,y)

# DecisionTree classifier plot
from sklearn.externals.six import StringIO
from sklearn.tree import export_graphviz
import pydot

features = ["Jet Mass", "tau 3/tau 2"]
classes = ["background","signal"]

dot_data = StringIO() 
export_graphviz(clf, out_file=dot_data,
                feature_names=features,  
                class_names=classes,  
                filled=True, rounded=True,  
                special_characters=True)
graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
graph.write_pdf("nsub.pdf")

plot_colors = "br"
plot_step = 0.02
class_names = "AB"

plt.figure(figsize=(10, 5))

# Plot the decision boundaries
plt.subplot(121)
x_min, x_max = X[:, 0].min(), X[:, 0].max()
y_min, y_max = X[:, 1].min(), X[:, 1].max()
xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                     np.arange(y_min, y_max, plot_step))

Z = bdt.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
plt.axis("tight")

labels = ["background","signal"]

# Plot the training points
for i, n, c in zip(range(2), class_names, plot_colors):
    idx = np.where(y == i)
    plt.scatter(X[idx, 0], X[idx, 1],
                c=c, cmap=plt.cm.Paired,
                label=labels[i],s=5.)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.legend(loc='upper right')
plt.xlabel('Jet Mass')
plt.ylabel(r'$\tau_{32}$')
plt.title('Decision Boundary')

# calculate the decision scores
twoclass_output = bdt.decision_function(X)

# convert the decision scores into roc curve
es, eb  = [], []
for c in np.arange(0,1.,0.01):
    es.append(float((twoclass_output[ y == 1] > c).sum())/len(twoclass_output[y == 1]))
    eb.append(float((twoclass_output[ y == 0] > c).sum())/len(twoclass_output[y == 0]))

# plot roc-curve    
plt.subplot(122)    
print np.c_[es,eb]
plt.scatter(es,eb)
plt.show()
sys.exit(0)

# Plot the two-class decision scores
plot_range = (twoclass_output.min(), twoclass_output.max())
plt.subplot(122)
for i, n, c in zip(range(2), class_names, plot_colors):
    plt.hist(twoclass_output[y == i],
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

#plt.tight_layout()
plt.subplots_adjust(wspace=0.35)
plt.show()
