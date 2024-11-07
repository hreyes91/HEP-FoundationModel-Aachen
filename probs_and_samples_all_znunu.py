import os




num_samples_list=[200000]
train_batch_size=100
num_const=190
trunc_list=[5000]

model_name='model_best.pt'
num_epochs_test=5

#Different for znunu
#WtoQQ has different bin tag check that
#list_of_jets=['TTBar','ZJetsToNuNu','HToBB','HToCC','HToGG','HToWW2Q1L','HToWW4Q','TTBarLep','WToQ','ZToQQ']
#list_of_jets=['TTBar','HToBB','HToCC','HToGG','HToWW2Q1L','HToWW4Q','TTBarLep','ZToQQ']

list_of_jets_probs_other=['TTBar','HToBB','HToGG','HToWW2Q1L','HToWW4Q','TTBarLep','ZToQQ']

list_of_jets=['ZJetsToNuNu']
for jet in list_of_jets:

    bin_tag='10M_'+jet
    
    test_dataset='/net/data_t2k/transformers-hep/JetClass/discretized/'+jet+'_test___10M_'+jet+'.h5'
    
    mother_dir='/net/data_t2k/transformers-hep/JetClass/All_models_for_OptClass/'
    #tag_oftrain=''+jet+'_run_testwall_10M'



    for num_samples in num_samples_list:
        for trunc in trunc_list:

                tag_foreval='test_eval_nsamples'+str(num_samples)
                
                tag_forsample='samples_nsamples'+str(num_samples)+'_trunc_'+str(trunc)
                save_dir_tag='some_results_nsamples'+str(num_samples)+'_trunc_'+str(trunc)


                model_path=mother_dir+'/all_const_403030_'+jet+'/'


                command_sample= 'python sample_jets_0.py --model_dir '+model_path+' --savetag '+str(tag_forsample)+' --num_samples '+str(num_samples)+' --num_const '+str(num_const)+' --trunc '+str(trunc)+' --batchsize '+str(train_batch_size)+' --model_name '+model_name
                print(command_sample)
                os.system(command_sample)
                
                command_eval='python evaluate_probabilities.py --model '+str(model_path)+str(model_name)+' --data '+str(test_dataset)+' --tag '+tag_foreval+' --num_const '+str(num_const)+' --num_events '+str(num_samples)
        
                os.system(command_eval)
                
                for jet_other in list_of_jets_probs_other:
                
                    test_dataset_other='/net/data_t2k/transformers-hep/JetClass/discretized/'+jet_other+'_test___10M_'+jet_other+'.h5'
                    tag_foreval_other='test_eval_other_nsamples'+str(num_samples)+'_jet_'+str(jet_other)


                    command_eval_other='python evaluate_probabilities.py --model '+str(model_path)+str(model_name)+' --data '+str(test_dataset_other)+' --tag '+tag_foreval_other+' --num_const '+str(num_const)+' --num_events '+str(num_samples)

                    os.system(command_eval_other)




