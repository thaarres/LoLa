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
       "f_rec",
       "m_rec",
       "dRopt",  
       "fatjet.M()",
       "fatjet.Pt()",
       "filtered.M()",
       "filtered.Pt()",
       "softdropped.M()",
       "softdropped.Pt()",
       "is_signal_new",
]

clf_names = [
    "Lola_Ref1",
    "BDT_9v_r1_filev7",
    "NNXd_min_5deg_sample_v7_v28_v28_vanilla"
]
 
clf_objects = {
}

for n in clf_names:
    if not n in clf_objects.keys():
        clf_objects[n] = n


pretty_names = ["Lola Ref 1", "MotherOfTagger", "DeepTop"]

line_styles = ["solid", ":", "-."] * 30


plt.clf()
plt.figure()

colors = ['darkorange', 'limegreen',"navy", "grey"]

dfs = {}
for pretty_name, clf_name in zip(pretty_names,clf_names):
    store = pandas.HDFStore("output_" + clf_name + ".h5")
    print clf_name, store.keys()
    dfs[clf_name] = store["all"]


    print dfs[clf_name].keys()

    AOC = roc_auc_score(dfs[clf_name]["is_signal_new"], dfs[clf_name]["sigprob_" + clf_name])
    print("AOC: {0}".format(AOC))
    
    fpr, tpr, _ = roc_curve(dfs[clf_name]["is_signal_new"], dfs[clf_name]["sigprob_" + clf_name])

    fpr[fpr < 0.0001] = 0.0001

    plt.plot(tpr, 1./fpr, color=colors.pop(0), lw=2, label='{0} (area = {1:.2f})'.format(pretty_name, AOC))

plt.xlim([0.05, 1.0])
plt.ylim([1, 10000])
plt.yscale('log')

#plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('True Positive Rate')
plt.ylabel('1 / False Positive Rate')
plt.legend(loc="lower left")
plt.savefig("roc.png".format(clf_name))



# # Correlation Plots 
#print "Getting base"
#
#infname = "/mnt/t3nfs01/data01/shome/gregor/DeepTop/dnn_template/test-img-et-5deg-v7-testonly.h5"
#store_base = pandas.HDFStore(infname)
#df_tmp = store_base.select("table",columns = brs)
#df_base = df_tmp.set_index(['entry','is_signal_new'], drop=False)
#
#for clf_name in clf_names:    
#    print "Adding", clf_name
#    tmp = dfs[clf_name].set_index(['entry','is_signal_new'])
#    df_base = df_base.join(tmp, how='inner', rsuffix = "from_" + clf_name)
#
#print "All files loaded!"
#
## Calculate extra variables
#df_base["tau32_sd"] = df_base["tau3_sd"].astype(float)/df_base["tau2_sd"].astype(float)
#df_base["tau32"]    = df_base["tau3"].astype(float)/df_base["tau2"].astype(float)
#
#
#plots = []
#
#
## Plot Classifier outputs
#for clf_name in clf_names:
#    plots.append(["sigprob_"+clf_name, [], 100,0,1, "sigprob_"+clf_name])
#
## Other properties of top
## Plot inclusive as well as in slices of DNN Output
#proto_plots = [
#    # ["softdropped.M()", 100,0,300, "mass_sd"],
#    # ["filtered.M()", 100,0,300, "mass_filt"],
#    # ["fatjet.M()", 100,0,300, "mass_ungroomed"],
#    # ["softdropped.Pt()", 100,0,500, "pt_sd"],
#    # ["filtered.Pt()", 100,0,500, "pt_filt"],
#    # ["fatjet.Pt()", 100,300,500, "pt_ungroomed"],
#    # ["tau32_sd", 100,0,1, "tau32_sd"],
#    # ["tau32", 100,0,1, "tau32"],
#    # ["f_rec",100,0,0.5,"frec"],
#    # ["m_rec",100,-1,300,"HTT mass"],        
#    # ["dRopt",100,0, 2, "HTT dRopt"],               
#    ["sigprob_BDT_9v_r1_filev7", 100,0, 1, "BDT ouput"],               
#    ["sigprob_NNXd_min_5deg_sample_v7_v28_v28_vanilla", 100,0, 1, "DNN ouput"],               
#]                                      
#for [variable, nbins, xmin, xmax, name] in proto_plots:
#    plots.append([variable, [], nbins, xmin, xmax, name])
#    plots.append([variable, [df_base["sigprob_BDT_9v_r1_filev7"] > 0.8], nbins, xmin, xmax, name + "_bdthi"])
#    plots.append([variable, [df_base["sigprob_BDT_9v_r1_filev7"] < 0.2], nbins, xmin, xmax, name + "_bdtlo"])
#    plots.append([variable, [df_base["sigprob_NNXd_min_5deg_sample_v7_v28_v28_vanilla"] > 0.8], nbins, xmin, xmax, name + "_dnnhi"])
#    plots.append([variable, [df_base["sigprob_NNXd_min_5deg_sample_v7_v28_v28_vanilla"] < 0.2], nbins, xmin, xmax, name + "_dnnlo"])
#
#
## Make all plots
#for plot in plots:
#                     
#    [variable, cuts, nbins, xmin, xmax, name] = plot
#          
#    cut_sig = reduce(lambda x,y:x&y,cuts + [(df_base["is_signal_new"] == 1)])
#    cut_bkg = reduce(lambda x,y:x&y,cuts + [(df_base["is_signal_new"] == 0)])
#
#    sig = df_base.loc[cut_sig,variable]
#    bkg = df_base.loc[cut_bkg,variable]
#    
#    plt.clf()
#    counts_sig, bins_sig, bars_sig = plt.hist(sig, label="sig", bins=np.linspace(xmin,xmax,nbins), alpha=0.4, normed=True)
#    counts_bkg, bins_bkg, bars_bkg = plt.hist(bkg, label="bkg", bins=np.linspace(xmin,xmax,nbins), alpha=0.4, normed=True)
#
#    tfn = "newplots/dat/" + name  + ".dat"
#    tf = open(tfn, "w")
#    tf.write("bins_lo, bins_hi, counts_sig, counts_bkg\n")
#    for bl, bh, cs, cb in zip(bins_sig[:-1], bins_sig[1:], counts_sig, counts_bkg):
#        tf.write("{0}, {1}, {2}, {3}\n".format(bl,bh,cs,cb))
#    tf.close()
#        
#    plt.xlabel(variable, fontsize=16)
#    plt.ylabel("Fraction of jets", fontsize=16)        
#    plt.legend(loc=1)
#    plt.xlim(xmin,xmax)
#    plt.show()
#    plt.savefig("newplots/png/" + name)
#
#    
#
#
## And 2D Plots:
#for clf_name in clf_names:
#
#    print "Doing", clf_name
#
#    for var_ob in [["softdropped.M()", 100,0,300, "mass_sd"],
#                   ["filtered.M()", 100,0,300, "mass_filt"],
#                   ["fatjet.M()", 100,0,300, "mass_ungroomed"],
#                   ["softdropped.Pt()", 100,0,500, "pt_sd"],
#                   ["filtered.Pt()", 100,0,500, "pt_filt"],
#                   ["fatjet.Pt()", 100,300,500, "pt_ungroomed"],
#                   ["tau32_sd", 100,0,1, "tau32_sd"],
#                   ["tau32", 100,0,1, "tau32"],
#                   ["f_rec",100,0,0.5,"frec"],
#                   ["m_rec",100,-1,300,"HTT mass"],        
#                   ["dRopt",100,0, 2, "HTT dRopt"],               
#                   ["sigprob_BDT_9v_r1_filev7", 100,0, 1, "BDT ouput"],               
#                   ["sigprob_NNXd_min_5deg_sample_v7_v28_v28_vanilla", 100,0, 1, "DNN ouput"]]:
#
#        var = var_ob[0]
#        xmin = var_ob[2]
#        xmax = var_ob[3]
#
#        print var
#
#        cuts = [(df_base[var] >= xmin), (df_base[var] <= xmax)]
#        cut_sig = reduce(lambda x,y:x&y,cuts + [(df_base["is_signal_new"] == 1)])
#        cut_bkg = reduce(lambda x,y:x&y,cuts + [(df_base["is_signal_new"] == 0)])
#
#
#        prob_sig = df_base.loc[cut_sig,"sigprob_"+clf_name]
#        prob_bkg = df_base.loc[cut_bkg,"sigprob_"+clf_name]
#
#        var_sig = df_base.loc[cut_sig, var]
#        var_bkg = df_base.loc[cut_bkg, var]
#
#        name = var.replace("(","").replace(")","").replace(".","_")
#
#        plt.clf()
#        counts, xedges, yedges, Image =  plt.hist2d(var_sig, prob_sig, bins=100)
#        plt.show()  
#        plt.savefig("plots2d/png/"+clf_name + "-2d-" + name + "-sig.png")
#
#        tfn = "plots2d/dat/" +clf_name + "-2d-" + name + "-sig.dat"
#        tf = open(tfn, "w")
#        tf.write("xlow, xhigh, ylow, yhigh, counts\n")
#        for irow in range(len(counts)):
#            for icol in range(len(counts[0])):
#                tf.write("{0},{1},{2},{3},{4}\n".format(xedges[icol], 
#                                                      xedges[icol+1], 
#                                                      yedges[irow], 
#                                                      yedges[irow+1], 
#                                                      counts[irow][icol]))
#        tf.close()
#
#
#
#        plt.clf()
#        plt.hist2d(var_bkg, prob_bkg,bins=100,)
#        plt.show()   
#        plt.savefig("plots2d/png/"+clf_name + "-2d-" + name + "-bkg.png")
#
#        tfn = "plots2d/dat/" +clf_name + "-2d-" + name + "-bkg.dat"
#        tf = open(tfn, "w")
#        tf.write("xlow, xhigh, ylow, yhigh, counts\n")
#        for irow in range(len(counts)):
#            for icol in range(len(counts[0])):
#                tf.write("{0},{1},{2},{3},{4}\n".format(xedges[icol], 
#                                                      xedges[icol+1], 
#                                                      yedges[irow], 
#                                                      yedges[irow+1], 
#                                                      counts[irow][icol]))
#        tf.close()
#
#
 






