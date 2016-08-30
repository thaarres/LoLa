#######################################
# Imports
########################################

from TrainClassifiersBase import *


########################################
# Configuration
########################################

brs = ["entry", 
       "tau2",
       "tau3",       
       "tau2_sd",
       "tau3_sd",       
       "fatjet.M()",
       "fatjet.Pt()",
       "filtered.M()",
       "filtered.Pt()",
       "softdropped.M()",
       "softdropped.Pt()",
       "is_signal_new",
]


clf_names = [
    "NNXd_model10",
    "NNXd_model1_et-on-et",
    "NNXd_model1_pt-on-pt",

    "NNXd_model1_et",
    "NNXd_model1_pt",

    "NNXd_model10-on-et",
    "NNXd_model1_pt-on-et", 

    "NNXd_model10-on-pt" ,            
    "NNXd_model1_et-on-pt",                   
]

pretty_names = [
    "Train: E, Test: E",
    "Train: ET, Test: ET",
    "Train: PT, Test: PT",

    "Train ET, Test: E",
    "Train PT, Test E",

    "Train E, Test ET",
    "Train PT, Test ET ", 

    "Train E, Test PT" ,            
    "Train ET, Test PT",                   
]

line_styles = ["solid", "solid", "solid", "--","--", "--","--","--", "--"]


dfs = {}
for clf_name in clf_names:
    store = pandas.HDFStore("output_" + clf_name + ".h5")

    dfs[clf_name] = store["all"]

#for clf_name in clf_names_on_pt + clf_names_on_et:
#    dfs[clf_name] = dfs[clf_name].reset_index()

#infname_e = "/mnt/t3nfs01/data01/shome/gregor/DeepTop/dnn_template/test_v2.h5"
#infname_et = "/mnt/t3nfs01/data01/shome/gregor/DeepTop/dnn_template/test-et.h5"
#infname_pt = "/mnt/t3nfs01/data01/shome/gregor/DeepTop/dnn_template/test-pt.h5"

#store_base = pandas.HDFStore(infname_e)
#df_base_e = store_base.select("table",columns = brs)

#store_base_et = pandas.HDFStore(infname_et)
#df_base_et = store_base.select("table",columns = brs)
#df_base_et = df_base_et.reset_index()
#df_base_et = df_base_et[:len(dfs[clf_names_on_et[0]])]
#
#store_base_pt = pandas.HDFStore(infname_pt)
#df_base_pt = store_base_pt.select("table",columns = brs)
#df_base_pt = df_base_pt.reset_index()
#df_base_pt = df_base_pt[:len(dfs[clf_names_on_pt[0]])]

#dfs["on_e"]  = pandas.concat( [dfs[name] for name in clf_names_on_e] + [df_base_e],   axis=1)
#dfs["on_et"] = pandas.concat( [dfs[name] for name in clf_names_on_et] + [df_base_et], axis=1)
#dfs["on_pt"] = pandas.concat( [dfs[name] for name in clf_names_on_pt] + [df_base_pt], axis=1)

# Make ROC plots
# Together
rocplot_multi([x.replace("-on-et","").replace("-on-pt","") for x in clf_names], 
              [dfs[x] for x in clf_names],
              pretty_names,
              line_styles,
)



#rocplot_multi([name.replace("_run_on_pt","") for name in clf_names_on_pt], dfs["on_pt"],"-on_pt")
#rocplot_multi([name.replace("_run_on_et","") for name in clf_names_on_et], dfs["on_et"],"-on_et")



sys.exit()


#print "All files loaded!"
#
## Calculate extra variables
#df_all["tau32_sd"] = df_all["tau3_sd"].astype(float)/df_all["tau2_sd"].astype(float)
#df_all["tau32"]    = df_all["tau3"].astype(float)/df_all["tau2"].astype(float)

    
plots = []

# Plot Classifier outputs
for clf_name in clf_names:
    plots.append(["sigprob_"+clf_name, [], 100,0,1, "sigprob_"+clf_name])

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
    #plots.append([variable, [df_all["sigprob"] > 0.8], nbins, xmin, xmax, name + "_hi"])
    #plots.append([variable, [df_all["sigprob"] > 0.4, df_all["sigprob"] < 0.6], nbins, xmin, xmax, name + "_md"])
    #plots.append([variable, [df_all["sigprob"] < 0.2], nbins, xmin, xmax, name + "_lo"])


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
    plt.savefig(name)


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
    plt.savefig(clf_name + "-2d-" + name + "-sig.png")

    plt.clf()
    plt.hexbin(var_bkg, prob_bkg)
    plt.show()   
    plt.savefig(clf_name + "-2d-" + name + "-bkg.png")



             

########################################
# Train/Load classifiers and make ROCs
########################################

#[clf.prepare() for clf in classifiers]
#analyze_multi(classifiers)
#eval_single(classifiers[0])

 






