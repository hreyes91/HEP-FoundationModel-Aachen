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
    



config_file='config_true_landscape.json'



sig_file='/home/home3/institut_thp/hreyes/Transformers/transformers_finke/datasets/TopQuarkTag/topquarktagging_data/discretized/all_top_40_30_30_forall.h5'
bg_file='/home/home3/institut_thp/hreyes/Transformers/transformers_finke/datasets/TopQuarkTag/topquarktagging_data/discretized/all_qcd_40_30_30_forall.h5'

n_events_list=[800000,500000,50000,5000]


for n_events in n_events_list:



    
    config_dict=OpenConfig(config_file)
    out_dir='logs/true_landscape_test7_noval_'+str(n_events)
    UpdateConfig(config_dict,sig_file,bg_file,out_dir,n_events)
    SaveNewConfig(config_dict,config_file)
    train(config_file)
    config_file_trained=out_dir+'/config.json'
    
    evaluate(config_file_trained)

