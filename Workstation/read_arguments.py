import os
import matplotlib.pyplot as plt
import pandas as pd




def extract_value():


    return


file_name='/Users/humbertosmac/Documents/work/Transformers/Transformers_finke/Workstation/test_results/full_test_qcd/_1/arguments.txt'
f = open(file_name, "r")
lines = f.readlines()

data_dict={'name_sufix':[],'dropout':[],'lr':[],'hidden_dim':[],'num_layers':[],'num_heads':[] }


for line in lines:
    if 'num_events' in line:
        line=line.replace(' ','')
        num_events=int(line.split('num_events')[-1])
        print(num_events)
