import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy import stats


def read_file(file_name):

    f = open(file_name, "r")
    lines = f.readlines()

    return lines

def extract_value(var,lines):

    for line in lines:
        if 'lr_' in line:
            continue
        if var in line:
            line=line.replace(' ','')
            line=line.replace('\n','')

            value=line.split(var)[-1]
    
    return value


main_dir='/net/data_t2k/transformers-hep/JetClass/Classification/classification_topvsqcd_part_pt_one_binning//'
dirs=os.listdir(main_dir)
val='epochs'
new_subdir='/test_50epochs/'
'''
for dir in dirs:

    if 'test_3' not in dir:
        continue
    
    file_name=main_dir+'/'+dir+'/arguments.txt'
    lines=read_file(file_name)
    value=int(extract_value(val,lines))
    
    if value==50:
        os.system('mv '+main_dir+'/'+dir+' '+main_dir+'/'+new_subdir+'/')
        print(dir)
    else:
        continue
    
    
'''
main_dir=main_dir+'/'+new_subdir

new_subdir='/test_50epochs/'

os.makedirs(main_dir+'/'+new_subdir,exist_ok=True)

dirs=os.listdir(main_dir)
for dir in dirs:

        if 'test_3' not in dir:
            continue
        
        files=os.listdir(main_dir+'/'+dir)
        print(files)
        for file in files:
            
            if 'roc_test' in file:
                print(dir)
                os.system('mv '+main_dir+'/'+dir+' '+main_dir+'/'+new_subdir+'/')
                continue


