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


def concat_and_save(df_1,df_2,out_file):

    df_all=pd.concat([df_1,df_2],axis=0)
    
    df_all.to_hdf(out_file, key="raw", mode="a", complevel=9)
    
    return


types=['test','train']

for type in types:

    print(type)
    input_file='/net/data_t2k/transformers-hep/JetClass/'+type+'/TTBar_'+type+'.h5'
    data_1,df_1=read_input(input_file)

    print(np.shape(data_1))

    input_file='/net/data_t2k/transformers-hep/JetClass/'+type+'/ZJetsToNuNu_'+type+'.h5'
    data_2,df_2=read_input(input_file)

    print(np.shape(data_2))

    out_file='/net/data_t2k/transformers-hep/JetClass/'+type+'/TTBar_ZJetsToNuNu_'+type+'.h5'
    concat_and_save(df_1,df_2,out_file)


    data_all,df_all=read_input(out_file)
    print(np.shape(data_all))
