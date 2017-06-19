from TrainClassifiers import main

params = {"input_path"              : "/scratch/snx3000/gregork/",
          "output_path"             : "/scratch/snx3000/gregork/output/",
          "output_path"             : "./",
          "inputs"                  : "constit_lola",
          "model_name"              : "LolaTest_17D",        
          "n_classes"               : 2,
          "signal_branch"           : "is_signal_new",
          "nb_epoch"                : 100,
          "batch_size"              : 1024,
          "name_train"              : "topconst-train-v3-resort.h5" ,
          "name_test"               : "topconst-test-v3-resort.h5",
          "train_poly"              : False,
          "train_offset"            : "none",
          "train_metric"            : False,
          "train_minmax"            : True,
          "do_3_metric"             : True,
          "use_angular_dr"          : False,
          "do_mult_p"               : 1,
          "mult_p"                  : 0,
          "regularize_weight"       : True,
          "train_regularize_weight" : False,
          "train_regularize_weight_target" : False,
          "lola_filters"            : 4,
          "n_lolas"                 : 1,
}

main(params)
