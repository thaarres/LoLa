from TrainClassifiers import main

params = {"input_path"              : "/scratch/snx3000/gregork/",
          "output_path"             : "/scratch/snx3000/gregork/output/",
          "output_path"             : "./",
          "inputs"                  : "constit_lola",
          "model_name"              : "LolaTest_0D",        
          "n_classes"               : 2,
          "signal_branch"           : "is_signal_new",
          "nb_epoch"                : 50,
          "batch_size"              : 1024,
          "name_train"              : "topconst-train-v2-resort.h5" ,
          "name_test"               : "topconst-test-v2-resort.h5",
          "train_poly"              : False,
          "train_offset"            : "diag",
          "train_metric"            : False,
          "train_minmax"            : False,
          "use_angular_dr"          : False,
          "do_mult_p"               : 0,
          "mult_p"                  : 0,
          "regularize_weight"       : False,
          "train_regularize_weight" : False,
          "lola_filters"            : 1,
          "n_lolas"                 : 1,
}

main(params)
