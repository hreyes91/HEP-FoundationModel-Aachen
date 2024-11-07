import os



test_dataset='Workstation/samples_codebook_1const.h5'
model_path='/net/data_ttk/hreyes/JetClass/OptClass/TTBar_models/TTBar_run_test__part_pt_1Mfromeach_403030_test_2_343QU3V/'
model_type='model_best.pt'
num_const=1
tag_foreval='test_eval_codebook_nconst1_endtokentrue'
num_events=39401

command_eval='python evaluate_probabilities.py --model '+str(model_path)+str(model_type)+' --data '+str(test_dataset)+' --tag '+tag_foreval+' --num_const '+str(num_const)+' --num_events '+str(num_events)+' --fixed_samples'
os.system(command_eval)
