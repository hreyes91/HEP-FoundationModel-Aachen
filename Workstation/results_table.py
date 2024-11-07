import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
from data_eval_helpers import make_continues,Make_Plots,LoadTrue,LoadSGenamples,GetHighLevel,Wasserstein_distance

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


def dict_to_frame_and_save(data_dict,data_dict_result,path):

    frame_param=pd.DataFrame(data_dict)
    frame_result=pd.DataFrame(data_dict_result)
    frame=pd.concat([frame_param,frame_result],axis=1)
    frame=frame.sort_values(by=['w_distance_pt'])
    frame.to_csv(path+'/results.txt',index=False)
    
    return


dir_name='/Users/humbertosmac/Documents/work/Transformers/Transformers_finke/test_results/TTBar/scan_1'
results=os.listdir(dir_name)

data_dict={'name_sufix':[],'dropout':[],'lr':[],'hidden_dim':[],'num_layers':[],'num_heads':[]}
data_dict_result={'w_distance_pt':[],'w_distance_mj':[] ,'w_distance_mul':[] }


bins_path_prefix='../preprocessing_bins/'
bin_tag='10M_TTBar'
pt_bins = np.load(bins_path_prefix+'pt_bins_'+bin_tag+'.npy')
eta_bins = np.load(bins_path_prefix+'eta_bins_'+bin_tag+'.npy')
phi_bins = np.load(bins_path_prefix+'phi_bins_'+bin_tag+'.npy')

n_test_samples=200000
discrete_truedata_filename='../../datasets/JetClass/discretized/test/TTBar_test___10M_TTBar.h5'
jets_true,ptj_true,mj_true=LoadTrue(discrete_truedata_filename,n_test_samples,pt_bins,eta_bins,phi_bins)
pt_true, eta_true,phi_true,mul_true=GetHighLevel(jets_true)

for result in results:
    print(result)
    file_name_samples=dir_name+'/'+result+'/samples__nsamples200000_trunc_5000.h5'
    try:
        jets,ptj,mj=LoadSGenamples(file_name_samples,pt_bins,eta_bins,phi_bins)
        
        pt, eta,phi,mul=GetHighLevel(jets)
        print('mul')
        print(mul)
        w_distance_pt=Wasserstein_distance(ptj_true,ptj)
        w_distance_mj=Wasserstein_distance(mj_true,mj)
        w_distance_mul=Wasserstein_distance(mul_true,mul)
        print(w_distance_mul)
        if w_distance_mj==None:
            w_distance_mj=9999
        data_dict_result.get('w_distance_pt').append(w_distance_pt)
        data_dict_result.get('w_distance_mj').append(w_distance_mj)
        data_dict_result.get('w_distance_mul').append(w_distance_mul)
        
    except:
        
        data_dict_result.get('w_distance_pt').append(9999)
        data_dict_result.get('w_distance_mj').append(9999)
        data_dict_result.get('w_distance_mul').append(9999)
   
    
  
    

    file_name_arguments=dir_name+'/'+result+'/arguments.txt'
    try:
        lines=read_file(file_name_arguments)
    except:
        print('arguments.txt not found in '+str(result)+' skipping')
        
        for var in data_dict.keys():
            data_dict.get(var).append('None')
        
        continue
    
    
    
    for var in data_dict.keys():
        value=extract_value(var,lines)
        data_dict.get(var).append(value)
    
dict_to_frame_and_save(data_dict,data_dict_result,dir_name)

