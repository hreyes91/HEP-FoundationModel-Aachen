
import os
import numpy as np
import h5py
import pandas as pd
import matplotlib.pyplot as plt
import random
import string
import time

def random_string():
    # initializing size of string
    N = 7
 
    # using random.choices()
    # generating random strings
    res = ''.join(random.choices(string.ascii_uppercase +
                             string.digits, k=N))
    # print result
    print("The generated random string : " + str(res))
    return str(res)
    
main_dir_classif='/net/data_t2k/transformers-hep/JetClass/OptClass/Classification/top_vs_qcd/n_samples/'
sig_path='/net/data_t2k/transformers-hep/JetClass/OptClass/TTBar_models/TTBar_run_test__part_pt_1Mfromeach_403030_test_2_343QU3V/samples__nsamples1000000_trunc_5000.h5'
bg_path='/net/data_t2k/transformers-hep/JetClass/OptClass/ZJetsToNuNu_models/ZJetsToNuNu_run_test__part_pt_1Mfromeach_403030_test_2_BU2IWA1/samples__nsamples1000000_trunc_5000.h5'
#list_of_jets=['TTBar','ZJetsToNuNu','HToBB','HToCC','HToGG','HToWW2Q1L','HToWW4Q','TTBarLep','WToQ','ZToQQ']
#sig_list=['TTBar_train___1Mfromeach_403030.h5']
#bg_list=['ZJetsToNuNu_train___1Mfromeach_403030.h5']
num_epochs_list=[3]
dropout_list=[0.0]
num_heads_list=[4]
num_layers_list=[8]
hidden_dim_list=[256]
batch_size_list=[100]
num_events_list=[1000]
num_const_list=[128]
lr_list=[.001]

tag_of_train='top_vs_qcd_classifier_part_pt_onebinning_1'
log_dir=main_dir_classif+tag_of_train


for num_events in num_events_list:
    for num_const in num_const_list:
        for batch_size in batch_size_list:
            for num_epochs in num_epochs_list:
                        for num_layers in num_layers_list:
                            for dropout in dropout_list:
                                for num_heads in num_heads_list:
                                    for lr in lr_list:
                                        for hidden_dim in hidden_dim_list:
                                        
                                            start_time = time.time()
                                            name_sufix=random_string()
                                            train_command='python train_classifier_new.py   --log_dir '+str(log_dir)+' --bg '+str(bg_path)+' --sig '+str(sig_path)+' --num_const '+str(num_const)+' --num_epochs '+str(num_epochs)+'  --lr '+str(lr)+' --batch_size '+str(batch_size)+' --num_events '+str(num_events)+' --dropout '+str(dropout)+' --num_heads '+str(num_heads)+' --num_layers '+str(num_layers)+' --hidden_dim '+str(hidden_dim)+' --name_sufix '+str(name_sufix)+' --fixed_samples'
                                            os.system(train_command)
                                            end_time = time.time()

elapsed_time = end_time - start_time
print(f"Time taken: {elapsed_time} seconds")
