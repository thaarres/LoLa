from TrainClassifiers import main

params = {"input_path" : "./",
          "output_path" : "./",
          "model_name" : "Htest_2",        
          "n_classes"  : 5,
          "signal_branch" : "class_new",
          "nb_epoch"   : 2,
          "name_train" : "deeph-train-v1-resort.h5" ,
          "name_test"  : "deeph-test-v1-resort.h5",
}

main(params)
