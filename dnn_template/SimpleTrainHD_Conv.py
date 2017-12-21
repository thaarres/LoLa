from TrainClassifiers import main

import sys



params = {"input_path"              : "/remote/gpu04/kasieczka/",
          "output_path"             : "./",
          "inputs"                  : "2d",
          "model_name"              : "HDConv",
          "nb_epoch"                : 200,
          "batch_size"              : 16,
          "n_features"              : 4,
          "name_train"              : "train-img-min-5deg-v8.h5",
          "name_test"               : "test-img-min-5deg-v8.h5",
          "name_val"                : "test-img-min-5deg-v8.h5",
}

main(params)
