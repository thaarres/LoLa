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

#    "BDT_10v_r1_filev5",

#    "BDT_5v_r1_filev7",
#    "BDT_6v_r1_filev7",
#    "BDT_8v_r1_filev7",

    "BDT_9v_r1_filev7",    
    "NNXd_min_5deg_sample_v7_v28_v28_vanilla",


#    "NNXd_min_5deg_sample_v7_v28_v28_vanilla_scale_0.75",
#    "NNXd_min_5deg_sample_v7_v28_v28_vanilla_scale_0.775",
#    "NNXd_min_5deg_sample_v7_v28_v28_vanilla_scale_0.8",
#    "NNXd_min_5deg_sample_v7_v28_v28_vanilla_scale_0.825",
#    "NNXd_min_5deg_sample_v7_v28_v28_vanilla_scale_0.85",
#    "NNXd_min_5deg_sample_v7_v28_v28_vanilla_scale_0.875",
#    "NNXd_min_5deg_sample_v7_v28_v28_vanilla_scale_0.9",
#    "NNXd_min_5deg_sample_v7_v28_v28_vanilla_scale_0.925",
#    "NNXd_min_5deg_sample_v7_v28_v28_vanilla_scale_0.95",
#    "NNXd_min_5deg_sample_v7_v28_v28_vanilla_scale_0.975",
#    "NNXd_min_5deg_sample_v7_v28_v28_vanilla_scale_1.0",
#    "NNXd_min_5deg_sample_v7_v28_v28_vanilla_scale_1.025",
#    "NNXd_min_5deg_sample_v7_v28_v28_vanilla_scale_1.05",
#    "NNXd_min_5deg_sample_v7_v28_v28_vanilla_scale_1.075",
#    "NNXd_min_5deg_sample_v7_v28_v28_vanilla_scale_1.1",
#    "NNXd_min_5deg_sample_v7_v28_v28_vanilla_scale_1.125",
#    "NNXd_min_5deg_sample_v7_v28_v28_vanilla_scale_1.15",
#    "NNXd_min_5deg_sample_v7_v28_v28_vanilla_scale_1.175",
#    "NNXd_min_5deg_sample_v7_v28_v28_vanilla_scale_1.2",
#    "NNXd_min_5deg_sample_v7_v28_v28_vanilla_scale_1.225",
#    "NNXd_min_5deg_sample_v7_v28_v28_vanilla_scale_1.25",


#    "NNXd_et_5deg_sample_v6_v23_v23_conv_nfeat_10",
#    "NNXd_et_5deg_sample_v6_v23_v23_conv_nfeat_6",
#    "NNXd_et_5deg_sample_v6_v23_v23_conv_size_2",
#    "NNXd_et_5deg_sample_v6_v23_v23_conv_size_6",
#    "NNXd_et_5deg_sample_v6_v23_v23_conv_size_8",
#    "NNXd_et_5deg_sample_v6_v23_v23_n_blocks_1",
#    "NNXd_et_5deg_sample_v6_v23_v23_n_blocks_3",
#    "NNXd_et_5deg_sample_v6_v23_v23_n_blocks_4",
#    "NNXd_et_5deg_sample_v6_v23_v23_n_conv_layers_3",
#    "NNXd_et_5deg_sample_v6_v23_v23_n_conv_layers_4",
#    "NNXd_et_5deg_sample_v6_v23_v23_n_conv_layers_5",
#    "NNXd_et_5deg_sample_v6_v23_v23_n_dense_layers_2",
#    "NNXd_et_5deg_sample_v6_v23_v23_n_dense_layers_4",
#    "NNXd_et_5deg_sample_v6_v23_v23_n_dense_nodes_128",
#    "NNXd_et_5deg_sample_v6_v23_v23_n_dense_nodes_32",
#    "NNXd_et_5deg_sample_v6_v23_v23_pool_size_0",
#    "NNXd_et_5deg_sample_v6_v23_v23_vanilla",



#    "NNXd_et_5deg_sample_v7_v30_v30_rnd_0p05",
#    "NNXd_et_5deg_sample_v7_v30_v30_rnd_0p05_105",
#    "NNXd_et_5deg_sample_v7_v30_v30_rnd_0p05_110",
#    "NNXd_et_5deg_sample_v7_v30_v30_rnd_0p05_120",
#    "NNXd_et_5deg_sample_v7_v30_v30_rnd_0p05_200",


#"NNXd_et_5deg_sample_v7_v33_v33_vanilla_scale_0.5",
#"NNXd_et_5deg_sample_v7_v33_v33_vanilla_scale_0.6",
#"NNXd_et_5deg_sample_v7_v33_v33_vanilla_scale_0.7",
#"NNXd_et_5deg_sample_v7_v33_v33_vanilla_scale_0.8",
#"NNXd_et_5deg_sample_v7_v33_v33_vanilla_scale_0.85",
#"NNXd_et_5deg_sample_v7_v33_v33_vanilla_scale_0.9",
#"NNXd_et_5deg_sample_v7_v33_v33_vanilla_scale_0.95",
#"NNXd_et_5deg_sample_v7_v33_v33_vanilla_scale_0.97",
#"NNXd_et_5deg_sample_v7_v33_v33_vanilla_scale_1.0",
#"NNXd_et_5deg_sample_v7_v33_v33_vanilla_scale_1.03",
#"NNXd_et_5deg_sample_v7_v33_v33_vanilla_scale_1.05",
#"NNXd_et_5deg_sample_v7_v33_v33_vanilla_scale_1.1",
#"NNXd_et_5deg_sample_v7_v33_v33_vanilla_scale_1.15",
#"NNXd_et_5deg_sample_v7_v33_v33_vanilla_scale_1.2",
#"NNXd_et_5deg_sample_v7_v33_v33_vanilla_scale_1.3",
#"NNXd_et_5deg_sample_v7_v33_v33_vanilla_scale_1.4",
#"NNXd_et_5deg_sample_v7_v33_v33_vanilla_scale_1.5",
#"NNXd_et_5deg_sample_v7_v33_v33_vanilla_scale_1.6",
#"NNXd_et_5deg_sample_v7_v33_v33_vanilla_scale_1.7",
#"NNXd_et_5deg_sample_v7_v33_v33_vanilla_scale_1.8",
#"NNXd_et_5deg_sample_v7_v33_v33_vanilla_scale_1.9",
#"NNXd_et_5deg_sample_v7_v33_v33_vanilla_scale_2.0",


#    "NNXd_min_5deg_sample_v8_v32_v31_conv_nfeat_10_scale_1.0",
#    "NNXd_min_5deg_sample_v8_v32_v31_conv_nfeat_6_scale_1.0",
#    "NNXd_min_5deg_sample_v8_v32_v31_conv_size_2_scale_1.0",
#    "NNXd_min_5deg_sample_v8_v32_v31_conv_size_6_scale_1.0",
#    "NNXd_min_5deg_sample_v8_v32_v31_n_blocks_1_scale_1.0",
#    "NNXd_min_5deg_sample_v8_v32_v31_n_blocks_3_scale_1.0",
#    "NNXd_min_5deg_sample_v8_v32_v31_n_conv_layers_2_scale_1.0",
#    "NNXd_min_5deg_sample_v8_v32_v31_n_conv_layers_4_scale_1.0",
#    "NNXd_min_5deg_sample_v8_v32_v31_n_dense_layers_3_n_conv_layers_2_scale_1.0",
#    "NNXd_min_5deg_sample_v8_v32_v31_n_dense_layers_3_scale_1.0",
#    "NNXd_min_5deg_sample_v8_v32_v31_n_dense_layers_5_scale_1.0",
#    "NNXd_min_5deg_sample_v8_v32_v31_n_dense_nodes_128_scale_1.0",
#    "NNXd_min_5deg_sample_v8_v32_v31_n_dense_nodes_32_scale_1.0",
#    "NNXd_min_5deg_sample_v8_v32_v31_pool_size_0_scale_1.0",
#    "NNXd_min_5deg_sample_v8_v32_v31_pool_size_4_scale_1.0",
#    "NNXd_min_5deg_sample_v8_v32_v31_vanilla_scale_1.0",




#   "NNXd_min_5deg_sample_v6_v27_v27_vanilla",
#
#   "NNXd_min_5deg_sample_v8_v31_v31_conv_nfeat_10",
#   "NNXd_min_5deg_sample_v8_v31_v31_conv_nfeat_6",
#   "NNXd_min_5deg_sample_v8_v31_v31_conv_size_2",
#   "NNXd_min_5deg_sample_v8_v31_v31_conv_size_6",
#   "NNXd_min_5deg_sample_v8_v31_v31_n_blocks_1",
#   "NNXd_min_5deg_sample_v8_v31_v31_n_blocks_3",
#   "NNXd_min_5deg_sample_v8_v31_v31_n_conv_layers_2",
#   "NNXd_min_5deg_sample_v8_v31_v31_n_conv_layers_4",
#   "NNXd_min_5deg_sample_v8_v31_v31_n_dense_layers_3",
#   "NNXd_min_5deg_sample_v8_v31_v31_n_dense_layers_3_n_conv_layers_2",
#   "NNXd_min_5deg_sample_v8_v31_v31_n_dense_layers_5",
#   "NNXd_min_5deg_sample_v8_v31_v31_n_dense_nodes_128",
#   "NNXd_min_5deg_sample_v8_v31_v31_n_dense_nodes_32",
#   "NNXd_min_5deg_sample_v8_v31_v31_pool_size_0",
#   "NNXd_min_5deg_sample_v8_v31_v31_pool_size_4",
#   "NNXd_min_5deg_sample_v8_v31_v31_vanilla",
#



#    "BDT_2v_r1_filev7",

#    "NNXd_et_5deg_sample_v7_v30_v30_vanilla",
#    "NNXd_et_5deg_sample_v7_v30_v30_rnd_0p01",
#    "NNXd_et_5deg_sample_v7_v30_v30_rnd_0p02",
#    "NNXd_et_5deg_sample_v7_v30_v30_rnd_0p05",
#    "NNXd_et_5deg_sample_v7_v30_v30_rnd_0p1",
#
#    "NNXd_et_5deg_sample_v7_v30_v30_vanilla_120",
#    "NNXd_et_5deg_sample_v7_v30_v30_rnd_0p01_120",
#    "NNXd_et_5deg_sample_v7_v30_v30_rnd_0p02_120",
#    "NNXd_et_5deg_sample_v7_v30_v30_rnd_0p05_120",
#    "NNXd_et_5deg_sample_v7_v30_v30_rnd_0p1_120",

#    "NNXd_min_5deg_sample_v6_v27_v27_feat12_layers2",
#    "NNXd_min_5deg_sample_v6_v27_v27_feat12_layers2_pool0",
#    "NNXd_min_5deg_sample_v6_v27_v27_feat12_layers4",
#    "NNXd_min_5deg_sample_v6_v27_v27_feat12_layers4_pool0",
#    "NNXd_min_5deg_sample_v6_v27_v27_layers2",
#    "NNXd_min_5deg_sample_v6_v27_v27_layers2_pool0",
#    "NNXd_min_5deg_sample_v6_v27_v27_layers4",
#    "NNXd_min_5deg_sample_v6_v27_v27_layers4_pool0",
#    "NNXd_min_5deg_sample_v6_v27_v27_vanilla",


#    "NNXd_et_5deg_sample_v6_v23_v23_conv_nfeat_10",
#    "NNXd_et_5deg_sample_v6_v23_v23_conv_nfeat_6",
#    "NNXd_et_5deg_sample_v6_v23_v23_conv_size_2",
#    "NNXd_et_5deg_sample_v6_v23_v23_conv_size_6",
#    "NNXd_et_5deg_sample_v6_v23_v23_conv_size_8",
#    "NNXd_et_5deg_sample_v6_v23_v23_n_blocks_1",
#    "NNXd_et_5deg_sample_v6_v23_v23_n_blocks_3",
#    "NNXd_et_5deg_sample_v6_v23_v23_n_blocks_4",
#    "NNXd_et_5deg_sample_v6_v23_v23_n_conv_layers_3",
#    "NNXd_et_5deg_sample_v6_v23_v23_n_conv_layers_4",
#    "NNXd_et_5deg_sample_v6_v23_v23_n_conv_layers_5",
#    "NNXd_et_5deg_sample_v6_v23_v23_n_dense_layers_2",
#    "NNXd_et_5deg_sample_v6_v23_v23_n_dense_layers_4",
#    "NNXd_et_5deg_sample_v6_v23_v23_n_dense_nodes_128",
#    "NNXd_et_5deg_sample_v6_v23_v23_n_dense_nodes_32",
#    "NNXd_et_5deg_sample_v6_v23_v23_vanilla",    

#    "NNXd_min_5deg_sample_v7_v28_v28_vanilla",
#

    "NNXd_min_5deg_sample_v7_v28_v28_cutoff_0p1",
    "NNXd_min_5deg_sample_v7_v28_v28_cutoff_0p5",
    "NNXd_min_5deg_sample_v7_v28_v28_cutoff_1",
    "NNXd_min_5deg_sample_v7_v28_v28_cutoff_2",
    "NNXd_min_5deg_sample_v7_v28_v28_cutoff_3",
    "NNXd_min_5deg_sample_v7_v28_v28_cutoff_5",
    "NNXd_min_5deg_sample_v7_v28_v28_cutoff_10",
    "NNXd_min_5deg_sample_v7_v28_v28_cutoff_15",
    "NNXd_min_5deg_sample_v7_v28_v28_cutoff_20",
    "NNXd_min_5deg_sample_v7_v28_v28_cutoff_30",
    "NNXd_min_5deg_sample_v7_v28_v28_cutoff_50",
#



#    "NNXd_min_5deg_sample_v6_v26_v26_conv_nfeat_10",
#    "NNXd_min_5deg_sample_v6_v26_v26_conv_nfeat_12",
#    "NNXd_min_5deg_sample_v6_v26_v26_conv_nfeat_6",
#    "NNXd_min_5deg_sample_v6_v26_v26_conv_size_2",
#    "NNXd_min_5deg_sample_v6_v26_v26_conv_size_6",
#    "NNXd_min_5deg_sample_v6_v26_v26_n_blocks_1",
#    "NNXd_min_5deg_sample_v6_v26_v26_n_blocks_3",
#    "NNXd_min_5deg_sample_v6_v26_v26_n_conv_layers_2",
#    "NNXd_min_5deg_sample_v6_v26_v26_n_conv_layers_4",
#    "NNXd_min_5deg_sample_v6_v26_v26_n_conv_layers_5",
#    "NNXd_min_5deg_sample_v6_v26_v26_n_dense_layers_3",
#    "NNXd_min_5deg_sample_v6_v26_v26_n_dense_layers_5",
#    "NNXd_min_5deg_sample_v6_v26_v26_n_dense_nodes_128",
#    "NNXd_min_5deg_sample_v6_v26_v26_n_dense_nodes_32",
#    "NNXd_min_5deg_sample_v6_v26_v26_pool_size_0",
#    "NNXd_min_5deg_sample_v6_v26_v26_vanilla",




#    "NNXd_min_5deg_sample_v6_v24_v24_conv_nfeat_10",
#    "NNXd_min_5deg_sample_v6_v24_v24_conv_nfeat_6",
#    "NNXd_min_5deg_sample_v6_v24_v24_conv_size_2",
#    "NNXd_min_5deg_sample_v6_v24_v24_conv_size_6",
##    "NNXd_min_5deg_sample_v6_v24_v24_conv_size_8",
##    "NNXd_min_5deg_sample_v6_v24_v24_n_blocks_1",
#    "NNXd_min_5deg_sample_v6_v24_v24_n_blocks_3",
# #   "NNXd_min_5deg_sample_v6_v24_v24_n_blocks_4",
#    "NNXd_min_5deg_sample_v6_v24_v24_n_conv_layers_3",
#    "NNXd_min_5deg_sample_v6_v24_v24_n_conv_layers_4",
# #   "NNXd_min_5deg_sample_v6_v24_v24_n_conv_layers_5",
# #   "NNXd_min_5deg_sample_v6_v24_v24_n_dense_layers_2",
# #   "NNXd_min_5deg_sample_v6_v24_v24_n_dense_layers_4",
# #   "NNXd_min_5deg_sample_v6_v24_v24_n_dense_nodes_128",
# #   "NNXd_min_5deg_sample_v6_v24_v24_n_dense_nodes_32",
# #   "NNXd_min_5deg_sample_v6_v24_v24_pool_size_0",
#    "NNXd_min_5deg_sample_v6_v24_v24_vanilla",    

#    "NNXd_min_5deg_sample_v6_v25_v25_conv_nfeat_10",
#    "NNXd_min_5deg_sample_v6_v25_v25_conv_nfeat_6",
#    "NNXd_min_5deg_sample_v6_v25_v25_conv_size_2",
#    "NNXd_min_5deg_sample_v6_v25_v25_conv_size_6",
#    "NNXd_min_5deg_sample_v6_v25_v25_lr_0p01",
#    "NNXd_min_5deg_sample_v6_v25_v25_n_blocks_1",
#    "NNXd_min_5deg_sample_v6_v25_v25_n_blocks_3",
#    "NNXd_min_5deg_sample_v6_v25_v25_n_conv_layers_2",
#    "NNXd_min_5deg_sample_v6_v25_v25_n_conv_layers_4",
#    "NNXd_min_5deg_sample_v6_v25_v25_n_conv_layers_5",
#    "NNXd_min_5deg_sample_v6_v25_v25_n_dense_layers_2",
#    "NNXd_min_5deg_sample_v6_v25_v25_n_dense_layers_4",
#    "NNXd_min_5deg_sample_v6_v25_v25_n_dense_nodes_128",
#    "NNXd_min_5deg_sample_v6_v25_v25_n_dense_nodes_32",
#    "NNXd_min_5deg_sample_v6_v25_v25_pool_size_0",
#    "NNXd_min_5deg_sample_v6_v25_v25_vanilla",            




]
 
clf_objects = {
    "NNXd_et_5deg_sample_v7_v30_v30_vanilla_105" : "NNXd_et_5deg_sample_v7_v30_v30_vanilla" ,
    "NNXd_et_5deg_sample_v7_v30_v30_rnd_0p01_105": "NNXd_et_5deg_sample_v7_v30_v30_rnd_0p01",
    "NNXd_et_5deg_sample_v7_v30_v30_rnd_0p02_105": "NNXd_et_5deg_sample_v7_v30_v30_rnd_0p02",
    "NNXd_et_5deg_sample_v7_v30_v30_rnd_0p05_105": "NNXd_et_5deg_sample_v7_v30_v30_rnd_0p05",
    "NNXd_et_5deg_sample_v7_v30_v30_rnd_0p1_105" : "NNXd_et_5deg_sample_v7_v30_v30_rnd_0p1" ,


    "NNXd_et_5deg_sample_v7_v30_v30_vanilla_110" : "NNXd_et_5deg_sample_v7_v30_v30_vanilla" ,
    "NNXd_et_5deg_sample_v7_v30_v30_rnd_0p01_110": "NNXd_et_5deg_sample_v7_v30_v30_rnd_0p01",
    "NNXd_et_5deg_sample_v7_v30_v30_rnd_0p02_110": "NNXd_et_5deg_sample_v7_v30_v30_rnd_0p02",
    "NNXd_et_5deg_sample_v7_v30_v30_rnd_0p05_110": "NNXd_et_5deg_sample_v7_v30_v30_rnd_0p05",
    "NNXd_et_5deg_sample_v7_v30_v30_rnd_0p1_110" : "NNXd_et_5deg_sample_v7_v30_v30_rnd_0p1" ,

    "NNXd_et_5deg_sample_v7_v30_v30_vanilla_120" : "NNXd_et_5deg_sample_v7_v30_v30_vanilla" ,
    "NNXd_et_5deg_sample_v7_v30_v30_rnd_0p01_120": "NNXd_et_5deg_sample_v7_v30_v30_rnd_0p01",
    "NNXd_et_5deg_sample_v7_v30_v30_rnd_0p02_120": "NNXd_et_5deg_sample_v7_v30_v30_rnd_0p02",
    "NNXd_et_5deg_sample_v7_v30_v30_rnd_0p05_120": "NNXd_et_5deg_sample_v7_v30_v30_rnd_0p05",
    "NNXd_et_5deg_sample_v7_v30_v30_rnd_0p1_120" : "NNXd_et_5deg_sample_v7_v30_v30_rnd_0p1" ,


    "NNXd_et_5deg_sample_v7_v30_v30_vanilla_200"   : "NNXd_et_5deg_sample_v7_v30_v30_vanilla" ,
    "NNXd_et_5deg_sample_v7_v30_v30_rnd_0p01_200"  : "NNXd_et_5deg_sample_v7_v30_v30_rnd_0p01",
    "NNXd_et_5deg_sample_v7_v30_v30_rnd_0p02_200"  : "NNXd_et_5deg_sample_v7_v30_v30_rnd_0p02",
    "NNXd_et_5deg_sample_v7_v30_v30_rnd_0p05_200"  : "NNXd_et_5deg_sample_v7_v30_v30_rnd_0p05",
    "NNXd_et_5deg_sample_v7_v30_v30_rnd_0p1_200"   : "NNXd_et_5deg_sample_v7_v30_v30_rnd_0p1" ,


"NNXd_min_5deg_sample_v7_v28_v28_vanilla_scale_0.75" : "NNXd_min_5deg_sample_v7_v28_v28_vanilla",
"NNXd_min_5deg_sample_v7_v28_v28_vanilla_scale_0.775" : "NNXd_min_5deg_sample_v7_v28_v28_vanilla",
"NNXd_min_5deg_sample_v7_v28_v28_vanilla_scale_0.8" : "NNXd_min_5deg_sample_v7_v28_v28_vanilla",
"NNXd_min_5deg_sample_v7_v28_v28_vanilla_scale_0.825" : "NNXd_min_5deg_sample_v7_v28_v28_vanilla",
"NNXd_min_5deg_sample_v7_v28_v28_vanilla_scale_0.85" : "NNXd_min_5deg_sample_v7_v28_v28_vanilla",
"NNXd_min_5deg_sample_v7_v28_v28_vanilla_scale_0.875" : "NNXd_min_5deg_sample_v7_v28_v28_vanilla",
"NNXd_min_5deg_sample_v7_v28_v28_vanilla_scale_0.9" : "NNXd_min_5deg_sample_v7_v28_v28_vanilla",
"NNXd_min_5deg_sample_v7_v28_v28_vanilla_scale_0.925" : "NNXd_min_5deg_sample_v7_v28_v28_vanilla",
"NNXd_min_5deg_sample_v7_v28_v28_vanilla_scale_0.95" : "NNXd_min_5deg_sample_v7_v28_v28_vanilla",
"NNXd_min_5deg_sample_v7_v28_v28_vanilla_scale_0.975" : "NNXd_min_5deg_sample_v7_v28_v28_vanilla",
"NNXd_min_5deg_sample_v7_v28_v28_vanilla_scale_1.0" : "NNXd_min_5deg_sample_v7_v28_v28_vanilla",
"NNXd_min_5deg_sample_v7_v28_v28_vanilla_scale_1.025" : "NNXd_min_5deg_sample_v7_v28_v28_vanilla",
"NNXd_min_5deg_sample_v7_v28_v28_vanilla_scale_1.05" : "NNXd_min_5deg_sample_v7_v28_v28_vanilla",
"NNXd_min_5deg_sample_v7_v28_v28_vanilla_scale_1.075" : "NNXd_min_5deg_sample_v7_v28_v28_vanilla",
"NNXd_min_5deg_sample_v7_v28_v28_vanilla_scale_1.1" : "NNXd_min_5deg_sample_v7_v28_v28_vanilla",
"NNXd_min_5deg_sample_v7_v28_v28_vanilla_scale_1.125" : "NNXd_min_5deg_sample_v7_v28_v28_vanilla",
"NNXd_min_5deg_sample_v7_v28_v28_vanilla_scale_1.15" : "NNXd_min_5deg_sample_v7_v28_v28_vanilla",
"NNXd_min_5deg_sample_v7_v28_v28_vanilla_scale_1.175" : "NNXd_min_5deg_sample_v7_v28_v28_vanilla",
"NNXd_min_5deg_sample_v7_v28_v28_vanilla_scale_1.2" : "NNXd_min_5deg_sample_v7_v28_v28_vanilla",
"NNXd_min_5deg_sample_v7_v28_v28_vanilla_scale_1.225" : "NNXd_min_5deg_sample_v7_v28_v28_vanilla",
"NNXd_min_5deg_sample_v7_v28_v28_vanilla_scale_1.25" : "NNXd_min_5deg_sample_v7_v28_v28_vanilla",



    "NNXd_min_5deg_sample_v8_v32_v31_conv_nfeat_10_scale_1.0"                     :     "NNXd_min_5deg_sample_v8_v32_v31_conv_nfeat_10",                    
    "NNXd_min_5deg_sample_v8_v32_v31_conv_nfeat_6_scale_1.0"                      :     "NNXd_min_5deg_sample_v8_v32_v31_conv_nfeat_6",                     
    "NNXd_min_5deg_sample_v8_v32_v31_conv_size_2_scale_1.0"                       :     "NNXd_min_5deg_sample_v8_v32_v31_conv_size_2",                      
    "NNXd_min_5deg_sample_v8_v32_v31_conv_size_6_scale_1.0"                       :     "NNXd_min_5deg_sample_v8_v32_v31_conv_size_6",                      
    "NNXd_min_5deg_sample_v8_v32_v31_n_blocks_1_scale_1.0"                        :     "NNXd_min_5deg_sample_v8_v32_v31_n_blocks_1",                       
    "NNXd_min_5deg_sample_v8_v32_v31_n_blocks_3_scale_1.0"                        :     "NNXd_min_5deg_sample_v8_v32_v31_n_blocks_3",                       
    "NNXd_min_5deg_sample_v8_v32_v31_n_conv_layers_2_scale_1.0"                   :     "NNXd_min_5deg_sample_v8_v32_v31_n_conv_layers_2",                  
    "NNXd_min_5deg_sample_v8_v32_v31_n_conv_layers_4_scale_1.0"                   :     "NNXd_min_5deg_sample_v8_v32_v31_n_conv_layers_4",                  
    "NNXd_min_5deg_sample_v8_v32_v31_n_dense_layers_3_n_conv_layers_2_scale_1.0"  :     "NNXd_min_5deg_sample_v8_v32_v31_n_dense_layers_3_n_conv_layers_2", 
    "NNXd_min_5deg_sample_v8_v32_v31_n_dense_layers_3_scale_1.0"                  :     "NNXd_min_5deg_sample_v8_v32_v31_n_dense_layers_3",                 
    "NNXd_min_5deg_sample_v8_v32_v31_n_dense_layers_5_scale_1.0"                  :     "NNXd_min_5deg_sample_v8_v32_v31_n_dense_layers_5",                 
    "NNXd_min_5deg_sample_v8_v32_v31_n_dense_nodes_128_scale_1.0"                 :     "NNXd_min_5deg_sample_v8_v32_v31_n_dense_nodes_128",                
    "NNXd_min_5deg_sample_v8_v32_v31_n_dense_nodes_32_scale_1.0"                  :     "NNXd_min_5deg_sample_v8_v32_v31_n_dense_nodes_32",                 
    "NNXd_min_5deg_sample_v8_v32_v31_pool_size_0_scale_1.0"                       :     "NNXd_min_5deg_sample_v8_v32_v31_pool_size_0",                      
    "NNXd_min_5deg_sample_v8_v32_v31_pool_size_4_scale_1.0"                       :     "NNXd_min_5deg_sample_v8_v32_v31_pool_size_4",                      
    "NNXd_min_5deg_sample_v8_v32_v31_vanilla_scale_1.0"                           :     "NNXd_min_5deg_sample_v8_v32_v31_vanilla",                          


}

for n in clf_names:
    if not n in clf_objects.keys():
        clf_objects[n] = n


pretty_names = [

#    "Old 10var",

#    "Softdrop",
  #  "HTT",
  #  "HTT + softdrop tau",
  #  "Softdrop + HTT",
  #  "DNN", 
   # "DNN (unpreprocessed)",

  #  "2var HTT"

#    "scale_0.75",
#    "scale_0.775",
#    "scale_0.8",
#    "scale_0.825",
#    "scale_0.85",
#    "scale_0.875",
#    "scale_0.9",
#    "scale_0.925",
#    "scale_0.95",
#    "scale_0.975",
#    "scale_1.0",
#    "scale_1.025",
#    "scale_1.05",
#    "scale_1.075",
#    "scale_1.1",
#    "scale_1.125",
#    "scale_1.15",
#    "scale_1.175",
#    "scale_1.2",
#    "scale_1.225",
#    "scale_1.25",



    "v28: Default Pre",

    #"nominal",
    #"1.05",
    #"1.10",
    #"1.20",
    #"2.00",

#
#    "conv_nfeat_10",
#    "conv_nfeat_6",
#    "conv_size_2",
#    "conv_size_6",
#    "conv_size_8",
#    "n_blocks_1",
#    "n_blocks_3",
#    "n_blocks_4",
#    "n_conv_layers_3",
#    "n_conv_layers_4",
#    "n_conv_layers_5",
#    "n_dense_layers_2",
#    "n_dense_layers_4",
#    "n_dense_nodes_128",
#    "n_dense_nodes_32",
#    "pool_size_0",
#    "vanilla",


#    "conv_nfeat_10",                    
#    "conv_nfeat_6",                     
#    "conv_size_2",                      
#    "conv_size_6",                      
#    "n_blocks_1",                       
#    "n_blocks_3",                       
#    "n_conv_layers_2",                  
#    "n_conv_layers_4",                  
#    "n_dense_layers_3_n_conv_layers_2", 
#    "n_dense_layers_3",                 
#    "n_dense_layers_5",                 
#    "n_dense_nodes_128",                
#    "n_dense_nodes_32",                 
#    "pool_size_0",                      
#    "pool_size_4",                      
#    "vanilla",                          






#    "scale_0.5",   
#    "scale_0.6",   
#    "scale_0.7",   
#    "scale_0.8",   
#    "scale_0.85",  
#    "scale_0.9",   
#    "scale_0.95",  
#    "scale_0.97",  
#    "scale_1.0",   
#    "scale_1.03",  
#    "scale_1.05",  
#    "scale_1.1",   
#    "scale_1.15",  
#    "scale_1.2",   
#    "scale_1.3",   
#    "scale_1.4",   
#    "scale_1.5",   
#    "scale_1.6",   
#    "scale_1.7",   
#    "scale_1.8",   
#    "scale_1.9",   
#    "scale_2.0",   


#   "v27: Last best unpre",
#
#   "conv_nfeat_10",
#   "conv_nfeat_6",
#   "conv_size_2",
#   "conv_size_6",
#   "n_blocks_1",
#   "n_blocks_3",
#   "n_conv_layers_2",
#   "n_conv_layers_4",
#   "n_dense_layers_3",
#   "n_dense_layers_3_n_conv_layers_2",
#   "n_dense_layers_5",
#   "n_dense_nodes_128",
#   "n_dense_nodes_32",
#   "pool_size_0",
#   "pool_size_4",
#   "vanilla",

#    "conv_nfeat_10",
#    "conv_nfeat_6",
#    "conv_size_2",
#    "conv_size_6",
#    "conv_size_8",
#    "n_blocks_1",
#    "n_blocks_3",
#    "n_blocks_4",
#    "n_conv_layers_3",
#    "n_conv_layers_4",
#    "n_conv_layers_5",
#    "n_dense_layers_2",
#    "n_dense_layers_4",
#    "n_dense_nodes_128",
#    "n_dense_nodes_32",
#    "vanilla",    



#    "v30_vanilla",
#    "v30_rnd_0p01",
#    "v30_rnd_0p02",
#    "v30_rnd_0p05",
#    "v30_rnd_0p1",
#
#    "vanilla_120",
#    "rnd_0p01_120",
#    "rnd_0p02_120",
#    "rnd_0p05_120",
#    "rnd_0p1_120",



#    "v23",


    "v28_cutoff_0p1",
    "v28_cutoff_0p5",
    "v28_cutoff_1",
    "v28_cutoff_2",
    "v28_cutoff_3",
    "v28_cutoff_5",
    "v28_cutoff_10",
    "v28_cutoff_15",
    "v28_cutoff_20",
    "v28_cutoff_30",
    "v28_cutoff_50",


#    "v27_feat12_layers2",
#    "v27_feat12_layers2_pool0",
#    "v27_feat12_layers4",
#    "v27_feat12_layers4_pool0",
#    "v27_layers2",
#    "v27_layers2_pool0",
#    "v27_layers4",
#    "v27_layers4_pool0",
#    "v27_vanilla",

#    "v26_vanilla",


    



#
#    "conv_nfeat_10",
#    "conv_nfeat_12",
#    "conv_nfeat_6",
#    "conv_size_2",
#    "conv_size_6",
#    "n_blocks_1",
#    "n_blocks_3",
#    "n_conv_layers_2",
#    "n_conv_layers_4",
#    "n_conv_layers_5",
#    "n_dense_layers_3",
#    "n_dense_layers_5",
#    "n_dense_nodes_128",
#    "n_dense_nodes_32",
#    "pool_size_0",
#    "vanilla",
#


#
#    "v23",
#
#    "conv_nfeat_10",
#    "conv_nfeat_6",
#    "conv_size_2",
#    "conv_size_6",
#    "n_blocks_3",
#    "n_conv_layers_3",
#    "n_conv_layers_4",
#    "vanilla",    
#

#    "conv_nfeat_10",      
#    "conv_nfeat_6",       
#    "conv_size_2",        
#    "conv_size_6",        
#    "lr_0p01",            
#    "n_blocks_1",         
#    "n_blocks_3",         
#    "n_conv_layers_2",    
#    "n_conv_layers_4",    
#    "n_conv_layers_5",    
#    "n_dense_layers_2",   
#    "n_dense_layers_4",   
#    "n_dense_nodes_128",  
#    "n_dense_nodes_32",   
#    "pool_size_0",        
#    "vanilla",            
]

line_styles = ["solid", ":", "-."] * 30


dfs = {}
for clf_name in clf_names:
    store = pandas.HDFStore("output_" + clf_name + ".h5")
    print clf_name, store.keys()
    dfs[clf_name] = store["all"]

# Make ROC plots
# Together
rocplot_multi([clf_objects[x] for x in clf_names], 
              [dfs[x] for x in clf_names],
              pretty_names,
              line_styles,
)


print "Getting base"

infname = "/mnt/t3nfs01/data01/shome/gregor/DeepTop/dnn_template/test-img-et-5deg-v7-testonly.h5"
store_base = pandas.HDFStore(infname)
df_tmp = store_base.select("table",columns = brs)
df_base = df_tmp.set_index(['entry','is_signal_new'], drop=False)

for clf_name in clf_names:    
    print "Adding", clf_name
    tmp = dfs[clf_name].set_index(['entry','is_signal_new'])
    df_base = df_base.join(tmp, how='inner', rsuffix = "from_" + clf_name)

print "All files loaded!"

# Calculate extra variables
df_base["tau32_sd"] = df_base["tau3_sd"].astype(float)/df_base["tau2_sd"].astype(float)
df_base["tau32"]    = df_base["tau3"].astype(float)/df_base["tau2"].astype(float)


plots = []


# Plot Classifier outputs
for clf_name in clf_names:
    plots.append(["sigprob_"+clf_name, [], 100,0,1, "sigprob_"+clf_name])

# Other properties of top
# Plot inclusive as well as in slices of DNN Output
proto_plots = [
    # ["softdropped.M()", 100,0,300, "mass_sd"],
    # ["filtered.M()", 100,0,300, "mass_filt"],
    # ["fatjet.M()", 100,0,300, "mass_ungroomed"],
    # ["softdropped.Pt()", 100,0,500, "pt_sd"],
    # ["filtered.Pt()", 100,0,500, "pt_filt"],
    # ["fatjet.Pt()", 100,300,500, "pt_ungroomed"],
    # ["tau32_sd", 100,0,1, "tau32_sd"],
    # ["tau32", 100,0,1, "tau32"],
    # ["f_rec",100,0,0.5,"frec"],
    # ["m_rec",100,-1,300,"HTT mass"],        
    # ["dRopt",100,0, 2, "HTT dRopt"],               
    ["sigprob_BDT_9v_r1_filev7", 100,0, 1, "BDT ouput"],               
    ["sigprob_NNXd_min_5deg_sample_v7_v28_v28_vanilla", 100,0, 1, "DNN ouput"],               
]                                      
for [variable, nbins, xmin, xmax, name] in proto_plots:
    plots.append([variable, [], nbins, xmin, xmax, name])
    plots.append([variable, [df_base["sigprob_BDT_9v_r1_filev7"] > 0.8], nbins, xmin, xmax, name + "_bdthi"])
    plots.append([variable, [df_base["sigprob_BDT_9v_r1_filev7"] < 0.2], nbins, xmin, xmax, name + "_bdtlo"])
    plots.append([variable, [df_base["sigprob_NNXd_min_5deg_sample_v7_v28_v28_vanilla"] > 0.8], nbins, xmin, xmax, name + "_dnnhi"])
    plots.append([variable, [df_base["sigprob_NNXd_min_5deg_sample_v7_v28_v28_vanilla"] < 0.2], nbins, xmin, xmax, name + "_dnnlo"])


# Make all plots
for plot in plots:
                     
    [variable, cuts, nbins, xmin, xmax, name] = plot
          
    cut_sig = reduce(lambda x,y:x&y,cuts + [(df_base["is_signal_new"] == 1)])
    cut_bkg = reduce(lambda x,y:x&y,cuts + [(df_base["is_signal_new"] == 0)])

    sig = df_base.loc[cut_sig,variable]
    bkg = df_base.loc[cut_bkg,variable]
    
    plt.clf()
    counts_sig, bins_sig, bars_sig = plt.hist(sig, label="sig", bins=np.linspace(xmin,xmax,nbins), alpha=0.4, normed=True)
    counts_bkg, bins_bkg, bars_bkg = plt.hist(bkg, label="bkg", bins=np.linspace(xmin,xmax,nbins), alpha=0.4, normed=True)

    tfn = "newplots/dat/" + name  + ".dat"
    tf = open(tfn, "w")
    tf.write("bins_lo, bins_hi, counts_sig, counts_bkg\n")
    for bl, bh, cs, cb in zip(bins_sig[:-1], bins_sig[1:], counts_sig, counts_bkg):
        tf.write("{0}, {1}, {2}, {3}\n".format(bl,bh,cs,cb))
    tf.close()
        
    plt.xlabel(variable, fontsize=16)
    plt.ylabel("Fraction of jets", fontsize=16)        
    plt.legend(loc=1)
    plt.xlim(xmin,xmax)
    plt.show()
    plt.savefig("newplots/png/" + name)

    


# And 2D Plots:
for clf_name in clf_names:

    print "Doing", clf_name

    for var_ob in [["softdropped.M()", 100,0,300, "mass_sd"],
                   ["filtered.M()", 100,0,300, "mass_filt"],
                   ["fatjet.M()", 100,0,300, "mass_ungroomed"],
                   ["softdropped.Pt()", 100,0,500, "pt_sd"],
                   ["filtered.Pt()", 100,0,500, "pt_filt"],
                   ["fatjet.Pt()", 100,300,500, "pt_ungroomed"],
                   ["tau32_sd", 100,0,1, "tau32_sd"],
                   ["tau32", 100,0,1, "tau32"],
                   ["f_rec",100,0,0.5,"frec"],
                   ["m_rec",100,-1,300,"HTT mass"],        
                   ["dRopt",100,0, 2, "HTT dRopt"],               
                   ["sigprob_BDT_9v_r1_filev7", 100,0, 1, "BDT ouput"],               
                   ["sigprob_NNXd_min_5deg_sample_v7_v28_v28_vanilla", 100,0, 1, "DNN ouput"]]:

        var = var_ob[0]
        xmin = var_ob[2]
        xmax = var_ob[3]

        print var

        cuts = [(df_base[var] >= xmin), (df_base[var] <= xmax)]
        cut_sig = reduce(lambda x,y:x&y,cuts + [(df_base["is_signal_new"] == 1)])
        cut_bkg = reduce(lambda x,y:x&y,cuts + [(df_base["is_signal_new"] == 0)])


        prob_sig = df_base.loc[cut_sig,"sigprob_"+clf_name]
        prob_bkg = df_base.loc[cut_bkg,"sigprob_"+clf_name]

        var_sig = df_base.loc[cut_sig, var]
        var_bkg = df_base.loc[cut_bkg, var]

        name = var.replace("(","").replace(")","").replace(".","_")

        plt.clf()
        counts, xedges, yedges, Image =  plt.hist2d(var_sig, prob_sig, bins=100)
        plt.show()  
        plt.savefig("plots2d/png/"+clf_name + "-2d-" + name + "-sig.png")

        tfn = "plots2d/dat/" +clf_name + "-2d-" + name + "-sig.dat"
        tf = open(tfn, "w")
        tf.write("xlow, xhigh, ylow, yhigh, counts\n")
        for irow in range(len(counts)):
            for icol in range(len(counts[0])):
                tf.write("{0},{1},{2},{3},{4}\n".format(xedges[icol], 
                                                      xedges[icol+1], 
                                                      yedges[irow], 
                                                      yedges[irow+1], 
                                                      counts[irow][icol]))
        tf.close()



        plt.clf()
        plt.hist2d(var_bkg, prob_bkg,bins=100,)
        plt.show()   
        plt.savefig("plots2d/png/"+clf_name + "-2d-" + name + "-bkg.png")

        tfn = "plots2d/dat/" +clf_name + "-2d-" + name + "-bkg.dat"
        tf = open(tfn, "w")
        tf.write("xlow, xhigh, ylow, yhigh, counts\n")
        for irow in range(len(counts)):
            for icol in range(len(counts[0])):
                tf.write("{0},{1},{2},{3},{4}\n".format(xedges[icol], 
                                                      xedges[icol+1], 
                                                      yedges[irow], 
                                                      yedges[irow+1], 
                                                      counts[irow][icol]))
        tf.close()


 






