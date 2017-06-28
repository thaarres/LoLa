from TrainClassifiers import main

params = {"input_path"              : "./",
          "output_path"             : "./",
          "inputs"                  : "constit_lola",
          "model_name"              : "Lola_Poly3a",        
          "nb_epoch"                : 200,
          "batch_size"              : 40,
          "name_train"              : "topconst-train-v2-resort.h5" ,
          "name_test"               : "topconst-test-v2-resort.h5",

          "train_poly"                     : False,
          "train_offset"                   : "only",
          "train_metric"                   : False,
          "train_minmax"                   : False,
          "lola_filters"                   : 1,
          "use_angular_dr"                 : True,
          "n_lolas"                        : 1,
          "do_mult_p"                      : 0,
          "mult_p"                         : 0,
          "regularize_weight"              : False,
          "train_regularize_weight"        : False,
          "train_regularize_weight_target" : False,
          "take_diff"                      : True,
}

main(params)