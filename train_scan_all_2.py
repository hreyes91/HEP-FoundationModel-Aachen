import os

#list_of_jets=['TTBar','ZJetsToNuNu','HToBB','HToCC','HToGG','HToWW2Q1L','HToWW4Q','TTBarLep','WToQ','ZToQQ']
list_of_jets=['ZJetsToNuNu']
num_const_list=[50]
num_epochs_list=[30]
lr_list=[.001]
lr_decay_list=[.000001]
num_events_list=[10000000]
dropout_list=[0.0]
num_heads_list=[4]
num_layers_list=[8]
num_bins_list=["41 31 31"]
weight_decay_list=[0.00001]
hidden_dim_list=[256]
batch_size_list=[200]

for jet in list_of_jets:

    data_path='/net/data_t2k/transformers-hep/JetClass/discretized/'+jet+'_train___10M_'+jet+'.h5'
    model_path='/net/data_t2k/transformers-hep/JetClass/'+jet+'_models/'+jet+'_run_testwall_10M'
    log_dir='/net/data_t2k/transformers-hep/JetClass/'+jet+'_models/'+jet+'_run_testwall_10M'
    output='linear'


    for num_events  in num_events_list:
        for num_const in num_const_list:
            for batch_size in batch_size_list:
                for num_bins in num_bins_list:
                    for num_epochs in num_epochs_list:
                        for lr_decay in lr_decay_list:
                            for weight_decay in weight_decay_list:
                                for num_layers in num_layers_list:
                                    for dropout in dropout_list:
                                        for num_heads in num_heads_list:
                                            for lr in lr_list:
                                                for hidden_dim in hidden_dim_list:
                                                    os.system('python train_2.py --data_path '+str(data_path)+' --model_path '+str(model_path)+' --log_dir '+str(log_dir)+'  --output '+str(output)+' --num_const '+str(num_const)+' --num_epochs '+str(num_epochs)+'  --lr '+str(lr)+' --lr_decay '+str(lr_decay)+' --batch_size '+str(batch_size)+' --num_events '+str(num_events)+' --dropout '+str(dropout)+' --num_heads '+str(num_heads)+' --num_layers '+str(num_layers)+' --num_bins '+str(num_bins)+' --weight_decay '+str(weight_decay)+' --hidden_dim '+str(hidden_dim)+' --end_token --start_token ')
                                        
                                        
                                    
                            
            
    
