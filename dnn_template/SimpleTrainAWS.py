
from TrainClassifiers import main


params = {"input_path"              : "./",
          "output_path"             : "./",
          "inputs"                  : "constit_lola",
          "model_name"              : "LoLaTest_Ref1_10const",        
          "nb_epoch"                : 100,
          "batch_size"              : 1024,
          "name_train"              : "topconst-train-v2-resort.h5" ,
          "name_test"               : "topconst-test-v2-resort.h5",
          "n_constit"               : 10,
}

main(params)
