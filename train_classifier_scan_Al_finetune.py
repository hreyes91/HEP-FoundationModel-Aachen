
import os
import numpy as np
import h5py
import pandas as pd
import matplotlib.pyplot as plt
import random
import string

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
    
main_dir_discrete='/net/data_ttk/hreyes/LHCO/LHCO_discrete/'

sig_list=['discrete_Weak-mix-Train-10000_1Mfromeach_403030.h5']
bg_list=['discrete_bg-N100-SR-Train_1Mfromeach_403030.h5']
num_epochs_list=[5]
dropout_list=[0.0]
num_heads_list=[4]
num_layers_list=[8]
hidden_dim_list=[256]
batch_size_list=[100]
num_events_list=[100000]
num_const_list=[128]
lr_list=[.001]
#num_events_val_max=500000

tag_of_train='LHCO_test_AL_1'
log_dir='/net/data_ttk/hreyes/LHCO/Classification_AL_finetune/'+tag_of_train
model_name='model_best.pt'
model_path_in='/net/data_ttk/hreyes/JetClass/OptClass/ZJetsToNuNu_models/ZJetsToNuNu_run_test__part_pt_1Mfromeach_403030_test_2_BU2IWA1/'
for sig in sig_list:
    for bg in bg_list:
        sig_path=main_dir_discrete+sig
        bg_path=main_dir_discrete+bg

        for num_events in num_events_list:
            '''
            if num_events<num_events_val_max:
                num_events_val=num_events
            else:
                num_events_val=num_events_val_max
            '''
            for num_const in num_const_list:
                for batch_size in batch_size_list:
                    for num_epochs in num_epochs_list:
                                for num_layers in num_layers_list:
                                    for dropout in dropout_list:
                                        for num_heads in num_heads_list:
                                            for lr in lr_list:
                                                for hidden_dim in hidden_dim_list:
                                                
                                                
                                                    name_sufix=random_string()
                                                    train_command='python train_classifier_AL_1.py   --log_dir '+str(log_dir)+' --bg '+str(bg_path)+' --sig '+str(sig_path)+' --num_const '+str(num_const)+' --num_epochs '+str(num_epochs)+'  --lr '+str(lr)+' --batch_size '+str(batch_size)+' --name_sufix '+str(name_sufix)+' --model_name '+str(model_name)+' --model_path_in '+str(model_path_in)+' --dropout '+str(dropout)
                                                    os.system(train_command)

