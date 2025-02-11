import numpy as np
import os
import h5py
from scipy import stats
import matplotlib.pyplot as plt

import data_eval_helpers2 as deh

#class
data='WToQQ'
n_events='_10M_events'
n_epochs='_30_epochs'
n_samples='_200k_samples'
path_to_plots='plots_'+data+n_events+n_epochs+n_samples

filename_samples='/net/data_ttk/koller/model_data_'+data+'_10M_train_ev_30_ep/samples_test.h5'
pt_bins=np.load('/net/data_ttk/hreyes/OneBin/preprocessing_bins/pt_bins_1Mfromeach_403030.npy')
eta_bins=np.load('/net/data_ttk/hreyes/OneBin/preprocessing_bins/eta_bins_1Mfromeach_403030.npy')
phi_bins=np.load('/net/data_ttk/hreyes/OneBin/preprocessing_bins/phi_bins_1Mfromeach_403030.npy')
n_samples_samples=200000

filename_disc_truedata='/net/data_ttk/koller/JetClass/discretized/'+data+'_train___1Mfromeach_403030.h5'
n_samples_true=200000

#get true and sampled data
jets,ptj,mj=deh.LoadSGenamples(filename_samples,pt_bins,eta_bins,phi_bins,n_samples_samples)
jets_true,ptj_true,mj_true=deh.LoadTrue(filename_disc_truedata,n_samples_true,pt_bins,eta_bins,phi_bins)
#plot distributions for pt, mj, eta phi
mul_samp,mul_true,pt_samp,pt_true=deh.Make_Plots(jets,pt_bins,eta_bins,phi_bins,mj,jets_true,ptj_true,mj_true,path_to_plots,n_events,n_epochs,n_samples,data)

#wd, ks
#w_distance,ks=deh.test_metrics(mj_true,mj)
#print('wassersteindistance: ',w_distance)
#print(ks)
