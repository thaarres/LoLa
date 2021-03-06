\from TrainClassifiers import main

import sys

NC = 15

params = {"input_path"              : "./",
          "output_path"             : "./",
          "inputs"                  : "constit_lola",
          "model_name"              : "Lola_Max_new_v0",
          "nb_epoch"                : 10,
          "batch_size"              : 1024,
          "n_constit"               : NC,
          "n_features"              : 4,
          "name_train"              : "topconst-val-v0_cand15_cFalse_vtxFalse-resort.h5".format(NC) ,
          "name_test"               : "topconst-val-v0_cand15_cFalse_vtxFalse-resort.h5".format(NC),
          "name_val"                : "topconst-val-v0_cand15_cFalse_vtxFalse-resort.h5".format(NC),
}

main(params)
