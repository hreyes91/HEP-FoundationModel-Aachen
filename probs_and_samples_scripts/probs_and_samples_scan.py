import os





test_dataset='/net/data_t2k/transformers-hep/JetClass/test/test_ZJetsToNuNu_.h5'
mother_dir='/net/data_t2k/transformers-hep/JetClass/ztonunu_models/'
model_type='model_last.pt'
tag_oftrain='zjettonunu_run_b'
num_samples=200
tag_foreval='test_eval_0'
tag_forsample='test_sample_0'
num_const=100

###For test samples
bg=test_dataset
bin_tag='10M_zjettonunu'
save_dir_tag='test_sample_test_0'
num_epochs_test=5

models_list=os.listdir(mother_dir)
for model in models_list:
    if tag_oftrain in model:
               
        model_path=mother_dir+'/'+model+'/'
        
        
        commad_eval='python evaluate_probabilities.py --model '+str(model_path)+str(model_type)+' --data '+str(test_dataset)+' --tag '+tag_foreval+' --num_const '+str(num_const)+' --num_events '+str(num_samples)
        
        
        os.system(commad_eval)
        commad_sample= 'python sample_jets.py --model_dir '+model_path+' --savetag '+str(tag_forsample)+' --num_samples '+str(num_samples)+' --num_const '+str(num_const)
        os.system(commad_sample)
        
        
        sg=model_path+'samples_'+tag_forsample+'.h5'
        save_dir=model_path+save_dir_tag
        command_test_sample='python test_samples.py --bg '+str(bg)+' --sig '+str(sg)+' --save_dir '+str(save_dir)+' --num_const '+str(num_const)+' --num_epochs '+str(num_epochs_test)+' -N '+str(num_samples)+' --bin_tag '+str(bin_tag)


        os.system(command_test_sample)


