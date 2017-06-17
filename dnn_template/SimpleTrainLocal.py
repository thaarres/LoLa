
from TrainClassifiers import main


params = {"input_path"              : "./",
          "output_path"             : "./",
          "inputs"                  : "constit_lola",
          "model_name"              : "LoLaTest_4",        
          "n_classes"               : 2,
          "signal_branch"           : "is_signal_new",
          "nb_epoch"                : 50,
          "batch_size"              : 32,
          "name_train"              : "topconst-train-v0-resort.h5" ,
          "name_test"               : "topconst-test-v0-resort.h5",
          "train_poly"              : False,
          "train_offset"            : "diag",
          "train_metric"            : False,
          "train_minmax"            : False,
          "use_angular_dr"          : True,
          "do_mult_p"               : 2,
          "mult_p"                  : -1,
          "regularize_weight"       : True,
          "train_regularize_weight" : True,
          "train_regularize_weight_target" : True,
          "do_3_metric"             : True,
          "lola_filters"            : 1,
          "n_lolas"                 : 1,
}

main(params)
