import numpy as np
import os
import h5py
from scipy import stats
import matplotlib.pyplot as plt

import data_eval_helpers2 as deh

data='TTBar'
n_events='_10M_events'
n_epochs='_30_epochs'
n_samples='_200k_samples'
path_to_plots='plots_'+data+n_events+n_epochs+n_samples

filename_samples_5000='/net/data_ttk/koller/model_data/model_data_'+data+'_10M_train_ev_30_ep/samples_test.h5'
filename_samples_1000='/net/data_ttk/koller/model_data/model_data_'+data+'_10M_train_ev_30_ep/samples_test_top_1000.h5'
filename_samples_2000='/net/data_ttk/koller/model_data/model_data_'+data+'_10M_train_ev_30_ep/samples_test_top_2000.h5'
filename_samples_2500='/net/data_ttk/koller/model_data/model_data_'+data+'_10M_train_ev_30_ep/samples_test_top_2500.h5'
filename_samples_3000='/net/data_ttk/koller/model_data/model_data_'+data+'_10M_train_ev_30_ep/samples_test_top_3000.h5'
filename_samples_4000='/net/data_ttk/koller/model_data/model_data_'+data+'_10M_train_ev_30_ep/samples_test_top_4000.h5'
pt_bins=np.load('/net/data_ttk/hreyes/OneBin/preprocessing_bins/pt_bins_1Mfromeach_403030.npy')
eta_bins=np.load('/net/data_ttk/hreyes/OneBin/preprocessing_bins/eta_bins_1Mfromeach_403030.npy')
phi_bins=np.load('/net/data_ttk/hreyes/OneBin/preprocessing_bins/phi_bins_1Mfromeach_403030.npy')
n_samples_samples=200000

filename_disc_truedata='/net/data_ttk/koller/JetClass/discretized/'+data+'_train___1Mfromeach_403030.h5'
n_samples_true=200000

#get true and sampled data
jets_true,ptj_true,mj_true=deh.LoadTrue(filename_disc_truedata,n_samples_true,pt_bins,eta_bins,phi_bins)
jets_5000,ptj_5000,mj_5000=deh.LoadSGenamples(filename_samples_5000,pt_bins,eta_bins,phi_bins,n_samples_samples)
jets_1000,ptj_1000,mj_1000=deh.LoadSGenamples(filename_samples_1000,pt_bins,eta_bins,phi_bins,n_samples_samples)
jets_2000,ptj_2000,mj_2000=deh.LoadSGenamples(filename_samples_2000,pt_bins,eta_bins,phi_bins,n_samples_samples)
jets_2500,ptj_2500,mj_2500=deh.LoadSGenamples(filename_samples_2500,pt_bins,eta_bins,phi_bins,n_samples_samples)
jets_3000,ptj_3000,mj_3000=deh.LoadSGenamples(filename_samples_3000,pt_bins,eta_bins,phi_bins,n_samples_samples)
jets_3000,ptj_3000,mj_4000=deh.LoadSGenamples(filename_samples_4000,pt_bins,eta_bins,phi_bins,n_samples_samples)

#plot distributions for pt, mj, eta, phi, mul
#mul_samp,mul_true,pt_samp,pt_true=deh.Make_Plots(jets_5000,pt_bins,eta_bins,phi_bins,mj_5000,jets_true,mj_true,path_to_plots)

jets_arr=[jets_true,jets_5000,jets_1000,jets_2000]
ptj_arr=[ptj_true,jets_5000,ptj_1000,ptj_2000]
mj_arr=[mj_1000,mj_2000,mj_2500,mj_3000,mj_4000,mj_5000,mj_true]

deh.comp_topk_plots(mj_arr,path_to_plots,data)