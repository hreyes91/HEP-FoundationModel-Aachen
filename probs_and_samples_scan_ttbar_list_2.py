import os

test_dataset='/net/data_t2k/transformers-hep/JetClass/discretized/TTBar_test___10M_TTBar.h5'
test_dataset_other='/net/data_t2k/transformers-hep/JetClass/discretized/ZJetsToNuNu_test___10M_ZJetsToNuNu.h5'
mother_dir='/net/data_t2k/transformers-hep/JetClass/TTBar_models/'
tag_oftrain='TTBar_run_testwall_10M'
num_samples_list=[200000]
train_batch_size=100
num_const=100
trunc_list=[5000]
###For test samples
bg=test_dataset
bin_tag='10M_TTBar'
model_name='model_best.pt'
num_epochs_test=5

models_list=os.listdir(mother_dir)

for num_samples in num_samples_list:
    for trunc in trunc_list:

        tag_foreval='test_eval_nsamples'+str(num_samples)
        tag_foreval_other='test_eval_other_nsamples'+str(num_samples)
        tag_forsample='samples_nsamples'+str(num_samples)+'_trunc_'+str(trunc)
        save_dir_tag='some_results_nsamples'+str(num_samples)+'_trunc_'+str(trunc)

        for model in models_list:
            if tag_oftrain in model:
                if ('_12' not in model) and ('_13' not in model):
                    continue
                model_path=mother_dir+'/'+model+'/'
                print(model)
        
                command_eval='python evaluate_probabilities.py --model '+str(model_path)+str(model_name)+' --data '+str(test_dataset)+' --tag '+tag_foreval+' --num_const '+str(num_const)+' --num_events '+str(num_samples)
        
        
                #os.system(command_eval)

                command_eval_other='python evaluate_probabilities.py --model '+str(model_path)+str(model_name)+' --data '+str(test_dataset_other)+' --tag '+tag_foreval_other+' --num_const '+str(num_const)+' --num_events '+str(num_samples)

                #os.system(command_eval_other)


                command_sample= 'python sample_jets_0.py --model_dir '+model_path+' --savetag '+str(tag_forsample)+' --num_samples '+str(num_samples)+' --num_const '+str(num_const)+' --trunc '+str(trunc)+' --batchsize '+str(train_batch_size)+' --model_name '+model_name
                print(command_sample)
                os.system(command_sample)
        
        
                sg=model_path+'samples_'+tag_forsample+'.h5'
                save_dir=model_path+save_dir_tag
                command_test_sample='python test_samples.py --bg '+str(bg)+' --sig '+str(sg)+' --save_dir '+str(save_dir)+' --num_const '+str(num_const)+' --num_epochs '+str(num_epochs_test)+' -N '+str(num_samples)+' --bin_tag '+str(bin_tag)


                os.system(command_test_sample)


