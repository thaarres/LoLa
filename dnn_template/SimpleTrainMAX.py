from TrainClassifiers import main

import sys

NC = 30

params = {"input_path"              : "/home/kasieczg/inputs/",
          "output_path"             : "./",
          "inputs"                  : "constit_lola",
          "model_name"              : "Lola_Max_baseline_v0",
          "nb_epoch"                : 200,
          "batch_size"              : 512,
          "n_constit"               : NC,
          "n_features"              : 4,
          "name_train"              : "topconst-train-v16_30nc-resort.h5".format(NC) ,
          "name_test"               : "topconst-test-v16_30nc-resort.h5".format(NC) ,
          "name_val"                : "topconst-val-v16_30nc-resort.h5".format(NC) ,
}

main(params)
