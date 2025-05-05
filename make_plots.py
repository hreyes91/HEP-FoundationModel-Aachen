import numpy as np
import os
import h5py
from scipy import stats
import matplotlib.pyplot as plt

import data_eval_helpers2 as deh

data_list=['TTBar']#,'HToGG',,'ZToQQ','TTBar'] #'HToGG','ZToNuNu'
n_events='_10M_events'
n_epochs='_30_epochs'
n_samples='_200k_samples'

pt_bins=np.load('/net/data_ttk/hreyes/OneBin/preprocessing_bins/pt_bins_1Mfromeach_403030.npy')
eta_bins=np.load('/net/data_ttk/hreyes/OneBin/preprocessing_bins/eta_bins_1Mfromeach_403030.npy')
phi_bins=np.load('/net/data_ttk/hreyes/OneBin/preprocessing_bins/phi_bins_1Mfromeach_403030.npy')

n_samples_samples=200000
n_samples_true=200000

for d in data_list:
    path_to_plots='plots_'+d+n_events+n_epochs+n_samples

    if d=='ZToNuNu':
        filename='/net/data_ttk/koller/JetClass/discretized/ZJetsToNuNu_train___1Mfromeach_403030.h5'
    else:
        filename='/net/data_ttk/koller/JetClass/discretized/'+d+'_train___1Mfromeach_403030.h5'
    jets_true,ptj_true,mj_true=deh.LoadTrue(filename,n_samples_true,pt_bins,eta_bins,phi_bins,key="discretized",sample=True)

    if d=='ZToNuNu':
        filename='/net/data_ttk/hreyes/JetClass/JetClass_pt_part/ZJetsToNuNu_train.h5'
    else:
        filename='/net/data_ttk/hreyes/JetClass/JetClass_pt_part/'+d+'_train.h5'
    jets_train,ptj_train,mj_train=deh.LoadJetClass(filename,nJets=10000000)

    filename='/net/data_ttk/koller/model_data/model_data_'+d+'_10M_train_ev_30_ep/samples_test.h5'
    jets_samp,ptj_samp,mj_samp=deh.LoadSGenamples(filename,pt_bins,eta_bins,phi_bins,n_samples_samples)


    mul_samp,mul_true,pt_samp,pt_true=deh.Make_Plots(jets_samp,pt_bins,eta_bins,phi_bins,mj_samp,jets_true,mj_true,path_to_plots,'true')
    mul_samp,mul_train,pt_samp,pt_train=deh.Make_Plots(jets_samp,pt_bins,eta_bins,phi_bins,mj_samp,jets_train,mj_train,path_to_plots,'train')
    