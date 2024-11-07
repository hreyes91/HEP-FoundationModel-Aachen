import os






mother_dir='/net/data_t2k/transformers-hep/JetClass/OptClass/ZJetsToNuNu_models/'
model_type='model_best.pt'
tag_oftrain='1Mfromeach_403030_'
num_samples_list=[200000,1000000]
train_batch_size=100
num_const_list=[128]
trunc_list=[5000]
###For test samples

bin_tag='1Mfromeach_403030'


models_list=os.listdir(mother_dir)

#for result in list_of_results:
#    model=tag_oftrain+'_'+str(result)
for model in models_list:
    
    if tag_oftrain not in model:
    #    if ('_6' not in model) and ('_10' not in model):
        continue

    for num_const in num_const_list:
        for num_samples in num_samples_list:
        
                test_dataset='net/data_t2k/transformers-hep/JetClass/OptClass/ZJetsToNuNu_models/ZJetsToNuNu_run_test__part_pt_1Mfromeach_403030_test_2_BU2IWA1/samples__nsamples'+str(num_samples)+'_trunc_5000.h5'
                test_dataset_other='/net/data_t2k/transformers-hep/JetClass/OptClass/TTBar_models//TTBar_run_test__part_pt_1Mfromeach_403030_test_2_343QU3V/samples__nsamples'+str(num_samples)+'_trunc_5000.h5'

                
                tag_foreval='test_eval_optclass_testset_nsamples'+str(num_samples)+'_numconst_'+str(num_const)
                tag_foreval_other='test_eval_optclass_testset_other_nsamples'+str(num_samples)+'_numconst_'+str(num_const)


                model_path=mother_dir+'/'+model+'/'
                print(model)
        
                command_eval='python evaluate_probabilities.py --model '+str(model_path)+str(model_type)+' --data '+str(test_dataset)+' --tag '+tag_foreval+' --num_const '+str(num_const)+' --num_events '+str(num_samples)+' --fixed_samples'
        
        
                os.system(command_eval)

                command_eval_other='python evaluate_probabilities.py --model '+str(model_path)+str(model_type)+' --data '+str(test_dataset_other)+' --tag '+tag_foreval_other+' --num_const '+str(num_const)+' --num_events '+str(num_samples)+' --fixed_samples'

                os.system(command_eval_other)


