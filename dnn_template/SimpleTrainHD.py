from TrainClassifiers import main

import sys



params = {"input_path"              : "/remote/gpu04/kasieczka/DeepTop/dnn_template/",
          "output_path"             : "./",
          "inputs"                  : "constit_lola",
          "model_name"              : "Lola_HD_{0}".format(sys.argv[1]),
          "nb_epoch"                : 200,
          "batch_size"              : 512,
          "n_constit"               : 40,
          "n_features"              : 8,
          "name_train"              : "top-train-v17a_cand40_cTrue_vtxTrue-resort-sl.h5",
          "name_test"               : "top-test-v17a_cand40_cTrue_vtxTrue-resort-sl.h5",
          "name_val"                : "top-val-v17a_cand40_cTrue_vtxTrue-resort-sl.h5",
}

main(params)
