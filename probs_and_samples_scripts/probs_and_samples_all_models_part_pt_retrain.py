import os




num_samples_list=[100000]
train_batch_size=100
num_const=128
trunc_list=[5000]

model_name='model_best.pt'
num_epochs_test=5

#Different for znunu
#WtoQQ has different bin tag check that
#list_of_jets=['TTBar','ZJetsToNuNu','HToBB','HToCC','HToGG','HToWW2Q1L','HToWW4Q','TTBarLep','WToQ','ZToQQ']
#list_of_jets=['TTBar','HToBB','HToCC','HToGG','HToWW2Q1L','HToWW4Q','TTBarLep','ZToQQ']
list_of_jets=['TTBar']


for jet in list_of_jets:

    bin_tag='40_30_30_pt_part_'+jet
    
    test_dataset='/net/data_t2k/transformers-hep/JetClass/discretized/'+jet+'_test___40_30_30_pt_part_'+jet+'.h5'
    test_dataset_other='/net/data_t2k/transformers-hep/JetClass/discretized/ZJetsToNuNu_test___40_30_30_pt_part_ZJetsToNuNu.h5'
    mother_dir='/net/data_t2k/transformers-hep/JetClass/TTBar_models/fine_tunning_pt_part/'
    #tag_oftrain=''+jet+'_run_testwall_10M'

    models_dirs=os.listdir(mother_dir)
    for model_dir in models_dirs:
        if 'TTBar_finetunefromQCD_const128_part_pt_4' not in model_dir:
            continue
        for num_samples in num_samples_list:
            for trunc in trunc_list:

                tag_foreval='test_eval_nsamples'+str(num_samples)
                tag_foreval_other='test_eval_other_nsamples'+str(num_samples)
                tag_forsample='samples_nsamples'+str(num_samples)+'_trunc_'+str(trunc)
                save_dir_tag='some_results_nsamples'+str(num_samples)+'_trunc_'+str(trunc)


                model_path=mother_dir+'/'+model_dir+'/'

        
                command_eval='python evaluate_probabilities.py --model '+str(model_path)+str(model_name)+' --data '+str(test_dataset)+' --tag '+tag_foreval+' --num_const '+str(num_const)+' --num_events '+str(num_samples)
        
        
                os.system(command_eval)

                command_eval_other='python evaluate_probabilities.py --model '+str(model_path)+str(model_name)+' --data '+str(test_dataset_other)+' --tag '+tag_foreval_other+' --num_const '+str(num_const)+' --num_events '+str(num_samples)

                os.system(command_eval_other)


                #command_sample= 'python sample_jets_1.py --model_dir '+model_path+' --savetag '+str(tag_forsample)+' --num_samples '+str(num_samples)+' --num_const '+str(num_const)+' --trunc '+str(trunc)+' --batchsize '+str(train_batch_size)+' --model_name '+model_name
                #print(command_sample)
                #os.system(command_sample)
        
                #bg=test_dataset
                #sg=model_path+'samples_'+tag_forsample+'.h5'
                #save_dir=model_path+save_dir_tag
                #command_test_sample='python test_samples.py --bg '+str(bg)+' --sig '+str(sg)+' --save_dir '+str(save_dir)+' --num_const '+str(num_const)+' --num_epochs '+str(num_epochs_test)+' -N '+str(num_samples)+' --bin_tag '+str(bin_tag)


                #os.system(command_test_sample)


