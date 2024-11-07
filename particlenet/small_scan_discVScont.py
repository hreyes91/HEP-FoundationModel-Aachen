import os

import json
from pprint import pprint


def OpenConfig(config_file_name):
    f=open(config_file_name)
    d = json.load(f)
    print(d)
    data=d.get('data')
    print(data)
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



def train_disc(config_file_disc):

    os.system('python train.py '+config_file_disc)

    return

def train_cont(config_file_cont):

    os.system('python train.py '+config_file_cont)

    return
    
def evaluate(config_file_trained):
    print('evaluating')
    os.system('python evaluate.py '+config_file_trained)
    
    return
    



config_file_disc='config_disc_scan.json'
config_file_cont='config_cont_scan.json'


prefix_znunu='/net/data_t2k/transformers-hep/JetClass/discretized/ZJetsToNuNu_test___10M_bins_tag_ZJetsToNuNu.h5'
prefix_top='/net/data_t2k/transformers-hep/JetClass/discretized/TTBar_test___10M_bins_tag_TTBar.h5'

bins_list=['80 60 60','20 15 15','10 7 7','5 3 3']
#bins_list=['40 30 30','20 15 15']


for bins in bins_list:
    bins_tag=bins.replace(' ','_')
    

    

    
    print(bins_tag)
    bg_file=prefix_znunu.replace('bins_tag',bins_tag)
    sig_file=prefix_top.replace('bins_tag',bins_tag)
    
    ####continuous
    
    #config_dict=OpenConfig(config_file_cont)
    #out_dir='logs/cont_scan_'+bins_tag
    #UpdateConfig(config_dict,sig_file,bg_file,out_dir)
    #SaveNewConfig(config_dict,config_file_name)
    #train_cont(config_file_cont)
    #config_file_trained=out_dir+'/config.json'
    #evaluate(config_file_trained)
    
    ####discrete
    
    config_dict=OpenConfig(config_file_disc)
    out_dir='logs/disc_scan_test_b_'+bins_tag+'_1'
    #UpdateConfig(config_dict,sig_file,bg_file,out_dir)
    #SaveNewConfig(config_dict,config_file_disc)

    config_file_trained=out_dir+'/config.json'
    #train_disc(config_file_disc)
    evaluate(config_file_trained)

