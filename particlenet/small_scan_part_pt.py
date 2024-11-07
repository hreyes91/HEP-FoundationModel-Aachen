import os

import json
from pprint import pprint


def OpenConfig(config_file_name):
    f=open(config_file_name)
    d = json.load(f)

    data=d.get('data')

    return d

def UpdateConfig(config_dict,sig_file,bg_file,out_dir):

    config_dict.get('data').update({'sig_file':sig_file})
    config_dict.get('data').update({'bg_file':bg_file})
    config_dict.get('logging').update({'logfolder':out_dir})
    
    print(config_dict)
    return config_dict
    
def SaveNewConfig(config_dict,config_file_name):

    save_file = open(config_file_name, "w")
    json.dump(config_dict, save_file,indent=6)
    save_file.close()


    return



def train(config_file):

    os.system('python train.py '+config_file)

    return

    
def evaluate(config_file_trained):
    print('evaluating')
    os.system('python evaluate.py '+config_file_trained)
    
    return
    



config_file='config_forgen_part_pt.json'



bg_file='/net/data_t2k/transformers-hep/JetClass/discretized/TTBar_test___40_30_30_pt_part_TTBar.h5'
sig_dirs_path='/net/data_t2k/transformers-hep/JetClass/TTBar_models/Part_pt_1/'

sig_dirs=os.listdir(sig_dirs_path)



for sig_dir in sig_dirs:
    if 'O0KHIRP' not in sig_dir:
        continue

    print(sig_dir)

    sig_file=sig_dirs_path+sig_dir+'/samples__nsamples100000_trunc_5000.h5'



    
    config_dict=OpenConfig(config_file)
    out_dir='logs/part_pt_gen_ttbar_50k_30epochs_10M_alldisc_'+str(sig_dir)
    UpdateConfig(config_dict,sig_file,bg_file,out_dir)
    SaveNewConfig(config_dict,config_file)
    train(config_file)
    config_file_trained=out_dir+'/config.json'
    
    evaluate(config_file_trained)

