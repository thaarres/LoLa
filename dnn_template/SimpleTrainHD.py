from TrainClassifiers import main

import sys

NC = int(sys.argv[1])

params = {"input_path"              : "/remote/gpu04/kasieczka/DeepTop/dnn_template/",
          "output_path"             : "./",
          "inputs"                  : "constit_lola",
          "model_name"              : "Lola_HD_{0}".format(sys.argv[2]),
          "nb_epoch"                : 200,
          "batch_size"              : 512,
          "n_constit"               : NC,
          "n_features"              : 4,
          "name_train"              : "topconst-train-v14_{0}nc-resort.h5".format(NC),
          "name_test"               : "topconst-test-v14_{0}nc-resort.h5".format(NC),
          "name_val"                : "topconst-val-v14_{0}nc-resort.h5".format(NC),
}

main(params)
