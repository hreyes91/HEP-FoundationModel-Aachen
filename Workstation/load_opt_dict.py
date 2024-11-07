import os
import numpy as np
import h5py
import pandas as pd
import matplotlib.pyplot as plt
import random
import string

import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np



from tqdm import tqdm


'''


path_to_sate_dict='../../test_results/Part_pt_1/TTBar_run_test__part_pt_const128_403030_3_O0KHIRP/opt_state_dict_best.pt'

checkpoint = torch.load(path_to_sate_dict,map_location=torch.device('cpu'))

state_dict=checkpoint['opt_state_dict_best']
print(state_dict.keys())
print(state_dict.get('state'))

for key in state_dict:
    print('hello')
    print(key, state_dict.keys())


for i in range(102):
    print(state_dict.get('state').get(i).get('exp_avg').shape)

'''
print('CLASSIFIER')
path_to_sate_dict='/Users/humbertosmac/Documents/work/Transformers/Transformers_finke/test_results/classifier/classification_part_pt/part_pt_2/top_vs_qcd_jetclass_classifier_part_pt_test_2_FAL6MQ3/opt_state_dict.pt'

checkpoint = torch.load(path_to_sate_dict,map_location=torch.device('cpu'))

state_dict=checkpoint['opt_state_dict']

size=len(state_dict.get('state').keys())

print('CLASSSFIER SIZE')
print(size)
for i in range(size):
    print(state_dict.get('state').get(i).get('exp_avg').shape)


############################



path_to_sate_dict='../../test_results/Part_pt_1/TTBar_run_test__part_pt_const128_403030_3_O0KHIRP/opt_state_dict_best.pt'

checkpoint = torch.load(path_to_sate_dict,map_location=torch.device('cpu'))

state_dict=checkpoint['opt_state_dict_best']


def GetLast2Layers(state_dict,last_keys):

    

    last2paramgroups=[]
    last2state=[]
    
    last2state.append(state_dict['state'][last_keys[0]])
    last2state.append(state_dict['state'][last_keys[1]])
    
    print(last2state)
    
    
    last2paramgroups.append(state_dict.get('param_groups')[0].get('params')[last_keys[0]])
    last2paramgroups.append(state_dict.get('param_groups')[0].get('params')[last_keys[1]])
    
    print(last2paramgroups)
    
    

    return last2paramgroups, last2state

def AddLayersToDict(filtered_sate_dict,last2state,last2paramgroups,last_keys):

    filtered_sate_dict.get('param_groups')[0].get('params').append(last2paramgroups[0])
    filtered_sate_dict.get('param_groups')[0].get('params').append(last2paramgroups[1])
    filtered_sate_dict['state'][last_keys[0]]=last2state[0]
    filtered_sate_dict['state'][last_keys[1]]=last2state[1]
    print(last2state[1])
    print(filtered_sate_dict['state'])
    print(filtered_sate_dict['param_groups'])
    
    return filtered_sate_dict
    
    
    

for key in state_dict:
    print('hello')
    print(key, state_dict.keys())

print(len(state_dict.get('state').keys()))
size=len(state_dict.get('state').keys())

print('gen SIZE')
print(size)


for i in range(size):
    print(state_dict.get('state').get(i).get('exp_avg').shape)




print('ONLY RELEVANT ONES')
print(state_dict.get('param_groups'))


state_keys = list(state_dict['state'].keys())
print(state_keys)
# Identify the last key
last_keys = state_keys[-2:]
print(last_keys)

last2paramgroups, last2state=GetLast2Layers(state_dict,last_keys)

# Remove the last entry
for last_key in reversed(last_keys):
    print(last_key)
    print(state_dict['state'][last_key])
    
    del state_dict['state'][last_key]

    state_dict.get('param_groups')[0].get('params').pop(last_key)

    
    
#print(state_keys)

#state_dict['state'].pop(102, None)

print(state_dict['state'].keys())


print(state_dict.get('param_groups')[0].get('params'))

torch.save(state_dict, 'modified_opt_state_dict.pt')


checkpoint_mod = torch.load('modified_opt_state_dict.pt',map_location=torch.device('cpu'))

print(checkpoint_mod.keys())

filtered_sate_dict=checkpoint_mod



all_again=AddLayersToDict(filtered_sate_dict,last2state,last2paramgroups,last_keys)
print('goodbye')
    
