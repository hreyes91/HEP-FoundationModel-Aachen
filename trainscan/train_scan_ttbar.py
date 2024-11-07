import os



data_path='/net/data_t2k/transformers-hep/JetClass/discretized/TTBar_train_top_10M_ttbar.h5'
model_path='/net/data_t2k/transformers-hep/JetClass/ttbar_models/ttbar_run_b_10M_const'
log_dir='/net/data_t2k/transformers-hep/JetClass/ttbar_models/ttbar_run_b_10M_const'
output='linear'



num_const_list=[100,50]
num_epochs_list=[50]
lr_list=[.0005]
lr_decay_list=[.000001]
num_events_list=[5000000]
dropout_list=[.1]
num_heads_list=[4]
num_layers_list=[8]
num_bins_list=["41 31 31"]
weight_decay_list=[0.00001]
hidden_dim_list=[256]
batch_size_list=[500]






for num_events  in num_events_list:
    for num_const in num_const_list:
            for num_bins in num_bins_list:
                    for num_epochs in num_epochs_list:
                        for lr_decay in lr_decay_list:
                            for weight_decay in weight_decay_list:
                                for num_layers in num_layers_list:
                                    for dropout in dropout_list:
                                        for num_heads in num_heads_list:
                                            for lr in lr_list:
                                                for hidden_dim in hidden_dim_list:
                                                    for batch_size in batch_size_list:
                                                        os.system('python train_2.py --data_path '+str(data_path)+' --model_path '+str(model_path)+' --log_dir '+str(log_dir)+'  --output '+str(output)+' --num_const '+str(num_const)+' --num_epochs '+str(num_epochs)+' --batch_size '+str(batch_size)+'  --lr '+str(lr)+' --lr_decay '+str(lr_decay)+' --num_events '+str(num_events)+' --dropout '+str(dropout)+' --num_heads '+str(num_heads)+' --num_layers '+str(num_layers)+' --num_bins '+str(num_bins)+' --weight_decay '+str(weight_decay)+' --hidden_dim '+str(hidden_dim)+' --end_token --start_token ')
                                        
                                        
                                    
                            
            
    
