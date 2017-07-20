from TrainClassifiers import main

import sys

NC = int(sys.argv[1])
SUFFIX = sys.argv[2]

params = {"input_path"              : "./",
          "output_path"             : "./",
          "inputs"                  : "constit_lola",
          "model_name"              : "Lola_AWS_Ref8_v11_{0}nc_{1}".format(NC,SUFFIX),        
          "nb_epoch"                : 3,
          "batch_size"              : 1024,
          "n_constit"               : NC,
          "n_features"              : 4,
          "name_train"              : "topconst-train-v11_{0}nc-resort.h5".format(NC) ,
          "name_test"               : "topconst-test-v11_{0}nc-resort.h5".format(NC),
          "name_val"                : "topconst-val-v11_{0}nc-resort.h5".format(NC),
}

main(params)
