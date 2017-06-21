from TrainClassifiers import main

params = {"input_path"              : "./",
          "output_path"             : "./",
          "inputs"                  : "constit_lola",
          "model_name"              : "Lola_Poly1",        
          "nb_epoch"                : 100,
          "batch_size"              : 1024,
          "name_train"              : "topconst-train-v3-resort.h5" ,
          "name_test"               : "topconst-test-v3-resort.h5",

          "train_poly"                     : True,
          "train_offset"                   : "full",
          "train_metric"                   : False,
          "train_minmax"                   : False,
          "lola_filters"                   : 1,
          "use_angular_dr"                 : False,
          "n_lolas"                        : 1,
          "do_mult_p"                      : 0,
          "mult_p"                         : 0,
          "regularize_weight"              : False,
          "train_regularize_weight"        : False,
          "train_regularize_weight_target" : False,
          "take_diff"                      : False, 
}

main(params)
