from TrainClassifiers import main

params = {"input_path"              : "./",
          "output_path"             : "./",
          "inputs"                  : "constit_lola",
          "model_name"              : "Lola_AWS_Ref8_v7",        
          "nb_epoch"                : 200,
          "batch_size"              : 1024,
          "n_constit"               : 20,
          "n_features"              : 4,
          "name_train"              : "topconst-train-v4-resort.h5" ,
          "name_test"               : "topconst-test-v4-resort.h5",


}

main(params)
