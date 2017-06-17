from TrainClassifiers import main

params = {"input_path"              : "/scratch/snx3000/gregork/",
          "output_path"             : "/scratch/snx3000/gregork/output/",
          "output_path"             : "./",
          "inputs"                  : "constit_lola",
          "model_name"              : "LolaTest_8D_15const",        
          "n_classes"               : 2,
          "signal_branch"           : "is_signal_new",
          "nb_epoch"                : 100,
          "batch_size"              : 1024,
          "name_train"              : "topconst-train-v3-resort.h5" ,
          "name_test"               : "topconst-test-v3-resort.h5",
          "train_poly"              : False,
          "train_offset"            : "full",
          "train_metric"            : False,
          "train_minmax"            : False,
          "use_angular_dr"          : True,
          "do_mult_p"               : 2,
          "mult_p"                  : -1,
          "regularize_weight"       : True,
          "train_regularize_weight" : True,
          "lola_filters"            : 2,
          "n_lolas"                 : 1,
}

main(params)
