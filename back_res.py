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
       ['test_top_1000','test_top_2000','test_top_2500','test_top_3000','test_top_4000','test_top_4500','test_top_4750','test','test_top_5250','test_top_5500','test_top_6000','test_10M_k5000','jetclass','true'],
       ['test_top_1000','test_top_2000','test_top_2500','test_top_3000','test_top_4000','test','test_top_5500','test_top_6000','test_1M_k5000','jetclass','true'],
       ['test_top_1000','test_top_2000','test_top_2500','test_top_3000','test_top_4000','test','test_top_5500','test_top_6000','test_top_6500','test_top_7000','test_top_8000','jetclass','true'],
       ['test_top_1000','test_top_2000','test_top_2500','test_top_3000','test_top_4000','test','test_top_5500','test_top_6000','test_top_6500','test_top_7000','test_top_8000','jetclass','true'],
       ['test_top_1000','test_top_1500','test_top_2000','test_top_2250','test_top_2500','test_top_2750','test_top_3000','test_top_4000','test','test_top_5500','test_top_6000','jetclass','true']
       ]
label_list=[
       ['k=1000','k=2000','k=2500','k=3000','k=4000','k=4500','k=4750','k=5000','k=5250','k=5500','k=6000','k=5000, 10M','JetClass','JetClass training data'],
       ['k=1000','k=2000','k=2500','k=3000','k=4000','k=5000','k=5500','k=6000','k=5000, 10M','JetClass','JetClass training data'],
       ['k=1000','k=2000','k=2500','k=3000','k=4000','k=5000','k=5500','k=6000','k=6500','k=7000','k=8000','JetClass','JetClass training data'],
       ['k=1000','k=2000','k=2500','k=3000','k=4000','k=5000','k=5500','k=6000','k=6500','k=7000','k=8000','JetClass','JetClass training data'],
       ['k=1000','k=1500','k=2000','k=2250','k=2500','k=2750','k=3000','k=4000','k=5000','k=5500','k=6000','JetClass','JetClass training data']
       ]
for d in range(len(data_list)):
    jets_arr=[]
    mj_arr=[]
    ptj_arr=[]
    data=data_list[d]
    k_val=k_val_list[d]
    label=label_list[d]
    print(data)
    #print(len(k_val),len(label))
    path_to_plots='plots_'+data+n_events+n_epochs+n_samples
    for k in k_val:
        if k=='true':
            if data=='ZToNuNu':
                filename='/net/data_ttk/koller/JetClass/discretized/ZJetsToNuNu_train___1Mfromeach_403030.h5'
            else:
                filename='/net/data_ttk/koller/JetClass/discretized/'+data+'_train___1Mfromeach_403030.h5'
            jets_true,ptj_true,mj_true=deh.LoadTrue(filename,n_samples_true,pt_bins,eta_bins,phi_bins)
            jets_arr.append(jets_true)
            ptj_arr.append(ptj_true)
            mj_arr.append(mj_true)
        elif k=='jetclass':
            filename=f'/net/data_ttk/hreyes/JetClass/JetClass_pt_part/{data}_train.h5'
            jets_con,ptj_con,mj_con=deh.LoadJetClass(filename)
            jets_arr.append(jets_con)
            ptj_arr.append(ptj_con)
            mj_arr.append(mj_con)
        else:
            filename='/net/data_ttk/koller/model_data/model_data_'+data+'_10M_train_ev_30_ep/samples_'+k+'.h5'
            jets,ptj,mj=deh.LoadSGenamples(filename,pt_bins,eta_bins,phi_bins,n_samples_samples)
            jets_arr.append(jets)
            ptj_arr.append(ptj)
            mj_arr.append(mj)
    #print(len(mj_arr),len(jets_arr),len(ptj_arr))
    ind_min_wd_mj,ind_min_wd_mul=deh.comp_topk_plots(mj_arr,jets_arr,path_to_plots,data,label)
    print(ind_min_wd_mj,ind_min_wd_mul)
    #ind_min=round((ind_min_wd_mul+ind_min_wd_mj)/2)
    #k_min=k_val[ind_min][-4:]

    #os.system('CUDA_VISIBLE_DEVICES=0 python sample_jets.py --model_dir /net/data_ttk/koller/model_data/model_data_'+data+'_10M_train_ev_30_ep --model_name model_best.pt --savetag test_10M_'+k_min+' --num_samples 200000 --num_const 128 --trunc '+k_min+' --batchsize 100')
    #jets_test,ptj_test,mj_test=deh.LoadSGenamples('/net/data_ttk/koller/model_data/model_data_'+data+'_10M_train_ev_30_ep/samples_test_top_'+k_min+'.h5',pt_bins,eta_bins,phi_bins,n_samples)
    #jets_test,ptj_test,mj_test=jets_arr[ind_min],ptj_arr[ind_min],mj_arr[ind_min]
    #mul_samp,mul_true,pt_samp,pt_true=deh.Make_Plots(jets_test,pt_bins,eta_bins,phi_bins,mj_test,jets_true,mj_true,path_to_plots)