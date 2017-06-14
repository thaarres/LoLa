from TrainClassifiers import main

params = {"input_path"    : "./",
          "output_path"   : "./",
          "inputs"        : "constit",
          "model_name"    : "ConstTest_0",        
          "n_classes"     : 2,
          "signal_branch" : "is_signal_new",
          "nb_epoch"      : 40,
          "batch_size"    : 100,
          "name_train"    : "topconst-train-v0-resort.h5" ,
          "name_test"     : "topconst-test-v0-resort.h5",
}

main(params)
