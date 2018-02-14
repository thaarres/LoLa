from TrainClassifiers import main

import sys



params = {"input_path"              : "/home/kasieczg/inputs/",
          "output_path"             : "./",
          "inputs"                  : "constit_lola",
          "model_name"              : "Lola_Max_baseline_v1",
          "nb_epoch"                : 200,
          "batch_size"              : 512,
          "n_constit"               : 40,
          "n_features"              : 4,
          "name_train"              : "topconst-train-v14_40nc-resort.h5",
          "name_test"               : "topconst-test-v14_40nc-resort.h5",
          "name_val"                : "topconst-val-v14_40nc-resort.h5",
}

main(params)
