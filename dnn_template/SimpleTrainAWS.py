from TrainClassifiers import main

params = {"input_path"              : "./",
          "output_path"             : "./",
          "inputs"                  : "constit_lola",
          "model_name"              : "Lola_AWS_Ref8_v7_40c_15o",        
          "nb_epoch"                : 200,
          "batch_size"              : 1024,
          "n_constit"               : 40,
          "n_features"              : 5,
          "name_train"              : "topconst-train-v7_40c-resort.h5" ,
          "name_test"               : "topconst-test-v7_40c-resort.h5",


}

main(params)
