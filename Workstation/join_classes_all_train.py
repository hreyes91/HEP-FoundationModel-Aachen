import os
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

def read_input(input_file,nJets=None):
    
        '''
        es = [f"E_{i}" for i in range(200)]
        px = [f"PX_{i}" for i in range(200)]
        py = [f"PY_{i}" for i in range(200)]
        pz = [f"PZ_{i}" for i in range(200)]
        cols = [item for sublist in zip(es, px, py, pz) for item in sublist]
        '''
        df = pd.read_hdf(
            input_file,
            key="raw",
            stop=nJets,
        )
        '''
        df = df[df["is_signal_new"] == class_label]
        df = df[cols]
        
        data = data.reshape((-1, 200, 4))
        '''
        print(df.head())
        data = df.to_numpy()
        return data,df


def concat_and_save(list_of_frames,out_file):

    df_all=pd.concat(list_of_frames,axis=0)
    
    df_all.to_hdf(out_file, key="raw", mode="a", complevel=9)
    
    return

list_of_jets=['TTBar','ZJetsToNuNu','HToBB','HToCC','HToGG','HToWW2Q1L','HToWW4Q','TTBarLep','WToQQ','ZToQQ']

n_per_jet=1000000
types=['train']

list_of_frames=[]
for type in types:
    for jet in list_of_jets:
        print(jet)
        input_file='/net/data_t2k/transformers-hep/JetClass/JetClass_pt_part/'+jet+'_'+type+'.h5'
        data_1,df_1=read_input(input_file)
    
        if n_per_jet != 'all':
            df_1=df_1.sample(n_per_jet)
    
        list_of_frames.append(df_1)

    out_file='/net/data_t2k/transformers-hep/JetClass/JetClass_pt_part/ALL_'+str(n_per_jet)+'fromeach_'+type+'.h5'
    concat_and_save(list_of_frames,out_file)


    data_all,df_all=read_input(out_file)
    print(np.shape(data_all))
