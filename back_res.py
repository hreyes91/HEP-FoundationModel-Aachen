import numpy as np
import os
import h5py
from scipy import stats
import matplotlib.pyplot as plt

import data_eval_helpers2 as deh

data_list=['WToQQ','ZToQQ']
n_events='_10M_events'
n_epochs='_30_epochs'
n_samples='_200k_samples'

pt_bins=np.load('/net/data_ttk/hreyes/OneBin/preprocessing_bins/pt_bins_1Mfromeach_403030.npy')
eta_bins=np.load('/net/data_ttk/hreyes/OneBin/preprocessing_bins/eta_bins_1Mfromeach_403030.npy')
phi_bins=np.load('/net/data_ttk/hreyes/OneBin/preprocessing_bins/phi_bins_1Mfromeach_403030.npy')

n_samples_samples=200000
n_samples_true=200000

k_val_list=[
       #['test_top_3000','test_top_4500','test','test_top_5500','test_top_6000','test_top_none','test_10M_k5000','true'],
       #['test_top_1500','test_top_2000','test_top_2250','test_top_2500','test','true'],
       #['test_top_4000','test','test_top_5500','test_top_6000','test_1M_k5000','true'],
       ['test_top_1000','test_top_2000','test_top_2500','test_top_3000','test_top_4000','test','test_top_5500','test_top_6000','test_top_6500','test_top_7000','test_top_8000','true'],
       ['test_top_1000','test_top_2000','test_top_2500','test_top_3000','test_top_4000','test','test_top_5500','test_top_6000','test_top_6500','test_top_7000','test_top_8000','true']
       ]
label_list=[
       #['k=3000','k=4500','k=5000','k=5500','k=6000','k none','k=5000, 10M','JetClass'],
       #['k=1500','k=2000','k=2250','k=2500','k=5000','JetClass'],
       #['k=4000','k=5000','k=5500','k=6000','k=5000, 1M','JetClass'],
       ['k=1500','k=2000','k=2500','k=3000','k=4000','k=5000','k=5500','k=6000','k=6500','k=7000','k=8000','JetClass'],
       ['k=1500','k=2000','k=2500','k=3000','k=4000','k=5000','k=5500','k=6000','k=6500','k=7000','k=8000','JetClass']
       ]
jets_arr=[]
mj_arr=[]
ptj_arr=[]
for d in range(len(data_list)):
    data=data_list[d]
    k_val=k_val_list[d]
    label=label_list[d]
    print(data)
    print(len(k_val),len(label))
    path_to_plots='plots_'+data+n_events+n_epochs+n_samples
    for k in k_val:
        if k=='true':
            if data=='ZToNuNu':
                filename='/net/data_ttk/koller/JetClass/discretized/ZJetsToNuNu_train___1Mfromeach_403030.h5'
            else:
                filename='/net/data_ttk/koller/JetClass/discretized/'+data+'_train___1Mfromeach_403030.h5'
            jets,ptj,mj=deh.LoadTrue(filename,n_samples_true,pt_bins,eta_bins,phi_bins)
        else:
            filename='/net/data_ttk/koller/model_data/model_data_'+data+'_10M_train_ev_30_ep/samples_'+k+'.h5'
            jets,ptj,mj=deh.LoadSGenamples(filename,pt_bins,eta_bins,phi_bins,n_samples_samples)
        jets_arr.append(jets)
        ptj_arr.append(ptj)
        mj_arr.append(mj)
    print(len(mj_arr),len(jets_arr),len(ptj_arr))
    ind_min_wd=deh.comp_topk_plots(mj_arr,jets_arr,path_to_plots,data,label)
    #k_min=k_val[ind_min_wd][-4:]
    print(ind_min_wd)#,k_min)

    #os.system('CUDA_VISIBLE_DEVICES=0 python sample_jets.py --model_dir /net/data_ttk/koller/model_data/model_data_'+data+'_10M_train_ev_30_ep --model_name model_best.pt --savetag test_10M_'+k_min+' --num_samples 200000 --num_const 128 --trunc '+k_min+' --batchsize 100')
    #jets_test,ptj_test,mj_test=deh.LoadSGenamples('/net/data_ttk/koller/model_data/model_data_HToGG_10M_train_ev_30_ep/samples_test_10M_'+k_min+'.h5',pt_bins,eta_bins,phi_bins,n_samples)
    #jets_test,ptj_test,mj_test=jets_arr[ind_min_wd],ptj_arr[ind_min_wd],mj_arr[ind_min_wd]
    #mul_samp,mul_true,pt_samp,pt_true=deh.Make_Plots(jets_test,pt_bins,eta_bins,phi_bins,mj_test,jets_test,mj_test,path_to_plots)
    #print('test last')