
from TrainClassifiers import main


params = {"input_path"    : "./",
          "output_path"   : "./",
          "inputs"        : "constit_lola",
          "model_name"    : "LoLaTest_1",        
          "n_classes"     : 2,
          "signal_branch" : "is_signal_new",
          "nb_epoch"      : 50,
          "batch_size"    : 32,
          "name_train"    : "topconst-train-v0-resort.h5" ,
          "name_test"     : "topconst-test-v0-resort.h5",
}

main(params)
