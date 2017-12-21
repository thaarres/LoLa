from TrainClassifiers import main

import sys

NC = int(sys.argv[1])

params = {"input_path"              : "/remote/gpu04/kasieczka/DeepTop/dnn_template/",
          "output_path"             : "./",
          "inputs"                  : "constit_lola",
          "model_name"              : "LolaHiggs_HD_{0}".format(sys.argv[2]),
          "nb_epoch"                : 100,
          "batch_size"              : 512,
          "n_constit"               : NC,
          "n_features"              : 4,
          "n_classes"               : 5,
          "name_train"              : "higgsconst-train-v1_cand{0}_cFalse_vtxFalse-resort-sl.h5".format(NC),
          "name_test"               : "higgsconst-test-v1_cand{0}_cFalse_vtxFalse-resort-sl.h5".format(NC),
          "name_val"                : "higgsconst-val-v1_cand{0}_cFalse_vtxFalse-resort-sl.h5".format(NC),
}

main(params)
