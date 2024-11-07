import os

import json
from pprint import pprint


def OpenConfig(config_file_name):
    f=open(config_file_name)
    d = json.load(f)

    data=d.get('data')

    return d

def UpdateConfig(config_dict,sig_file,bg_file,out_dir,n_events):

    config_dict.get('data').update({'sig_file':sig_file})
    config_dict.get('data').update({'bg_file':bg_file})
    config_dict.get('data').update({'n_jets':n_events})
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
    



config_file='config_true_jetclass.json'



sig_file='/net/data_t2k/transformers-hep/JetClass/JetClass_pt_part/TTBar_train.h5'
bg_file='/net/data_t2k/transformers-hep/JetClass/JetClass_pt_part/ZJetsToNuNu_train.h5'

n_events_list=[50,500,5000,50000,500000]


for n_events in n_events_list:



    
    config_dict=OpenConfig(config_file)
    out_dir='logs/true_jetclass_test3_'+str(n_events)
    UpdateConfig(config_dict,sig_file,bg_file,out_dir,n_events)
    SaveNewConfig(config_dict,config_file)
    train(config_file)
    config_file_trained=out_dir+'/config.json'
    
    evaluate(config_file_trained)

