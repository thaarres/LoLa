from TrainClassifiers import main

params = {"input_path"              : "/scratch/snx3000/gregork/",
          "output_path"             : "/scratch/snx3000/gregork/output/",
          "output_path"             : "./",
          "inputs"                  : "constit_lola",
          "model_name"              : "Lola_Ref1",        
          "nb_epoch"                : 100,
          "batch_size"              : 1024,
          "name_train"              : "topconst-train-v3-resort.h5" ,
          "name_test"               : "topconst-test-v3-resort.h5",
}

main(params)
