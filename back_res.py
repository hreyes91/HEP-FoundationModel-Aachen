import numpy as np
import os
import h5py
from scipy import stats
import matplotlib.pyplot as plt

import data_eval_helpers2 as deh

data='HToGG'
n_events='_10M_events'
n_epochs='_30_epochs'
n_samples='_200k_samples'
path_to_plots='plots_'+data+n_events+n_epochs+n_samples

pt_bins=np.load('/net/data_ttk/hreyes/OneBin/preprocessing_bins/pt_bins_1Mfromeach_403030.npy')
eta_bins=np.load('/net/data_ttk/hreyes/OneBin/preprocessing_bins/eta_bins_1Mfromeach_403030.npy')
phi_bins=np.load('/net/data_ttk/hreyes/OneBin/preprocessing_bins/phi_bins_1Mfromeach_403030.npy')

n_samples_samples=200000
n_samples_true=200000

k_val=['test_top_1000','test_top_2000','test_top_2500','test_top_3000','test_top_4000','test','test_top_5500','test_top_6000','true']
jets_arr=[]
mj_arr=[]
ptj_arr=[]
for k in k_val:
    if k=='true':
        filename='/net/data_ttk/koller/JetClass/discretized/'+data+'_train___1Mfromeach_403030.h5'
        jets,ptj,mj=deh.LoadTrue(filename,n_samples_true,pt_bins,eta_bins,phi_bins)
    else:
        filename='/net/data_ttk/koller/model_data/model_data_'+data+'_10M_train_ev_30_ep/samples_'+k+'.h5'
        jets,ptj,mj=deh.LoadSGenamples(filename,pt_bins,eta_bins,phi_bins,n_samples_samples)
    jets_arr.append(jets)
    ptj_arr.append(ptj)
    mj_arr.append(mj)

ind_min_wd=deh.comp_topk_plots(mj_arr,path_to_plots,data)
print(ind_min_wd,k_val[ind_min_wd])

mul_samp,mul_true,pt_samp,pt_true=deh.Make_Plots(jets_arr[ind_min_wd],pt_bins,eta_bins,phi_bins,mj_arr[ind_min_wd],jets_arr[-1],mj_arr[-1],path_to_plots)