import numpy as np
import pandas as pd
import torch
def OpenMainFile(main_file):
    tmp = pd.read_hdf(main_file, key="discretized", stop=None)

    return tmp

main_file='../../../Optimal_Classifier/LLRModels/ZJetsToNuNu_run_scan_10M_N1G96CW/samples__nsamples200000_trunc_5000.h5'

nconst_list=[80,60,40,20]


for nconst in nconst_list:

    base=np.ones((200000,nconst,3))
    base=torch.as_tensor(np.array(base),dtype=torch.int64)
    
    base_padding=np.ones((200000,nconst), dtype=bool)
    base_padding=torch.as_tensor(np.array(base_padding),dtype=torch.bool)


    
    extra=-1*np.ones((200000,100-nconst,3))

    extra_padding=np.zeros((200000, 100-nconst), dtype=bool)
    
    extra_padding=torch.as_tensor(np.array(extra_padding),dtype=torch.bool)
    print(extra_padding)
    print(np.shape(extra_padding))
    
    all_padding=torch.cat((base_padding,extra_padding),dim=1)
    print(all_padding)
    print(np.shape(all_padding))
    exit()
    
    #extra=torch.from_numpy()
    extra=torch.as_tensor(np.array(extra),dtype=torch.int64)
    print(extra)
    print(np.shape(extra))

    all=torch.cat((base,extra),dim=1)
    print(all)
    print(np.shape(all))

tmp=OpenMainFile(main_file)
print(tmp)
print(np.shape(tmp))
