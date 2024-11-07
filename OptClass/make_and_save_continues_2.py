import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy import stats
from data_eval_helpers import make_continues,LoadSGenamples, LoadTrue




def LoadBins():
    bins_path_prefix='/Users/humbertosmac/Documents/work/Transformers/Optimal_Classifier/LLRModels/preprocessing_bins/'
    pt_bins = np.load(bins_path_prefix+'pt_bins_40_30_30_pt_part_ZJetsToNuNu.npy')
    eta_bins = np.load(bins_path_prefix+'eta_bins_40_30_30_pt_part_ZJetsToNuNu.npy')
    phi_bins = np.load(bins_path_prefix+'phi_bins_40_30_30_pt_part_ZJetsToNuNu.npy')
    print('pt_bins')

    return pt_bins,eta_bins,phi_bins


def SaveSamples(continues_jets,mj,ptj,outpult_path):
    hf = h5py.File(outpult_path, 'w')
    hf.create_dataset('raw', data=continues_jets)
    hf.create_dataset('m_jet', data=mj)
    hf.create_dataset('ptj', data=ptj)
    
    hf.close()
    return


def ReadHDF(outpult_path):


    #tmp = pd.read_hdf(outpult_path, key="raw", stop=None)
    tmp = h5py.File(outpult_path, 'r')
    print(np.array(tmp.get('raw')))
    return

data_file='/Users/humbertosmac/Documents/work/Transformers/Transformers_finke/test_results/OPtclass/TTBar_run_test__part_pt_const128_403030_3_O0KHIRP/samples_samples_nsamples1000000_trunc_5000.h5'

outpult_path='/Users/humbertosmac/Documents/work/Transformers/Optimal_Classifier/TheSamples/OptimalClassifierSamples/top/raw/train_nsamples1M_trunc_5000.h5'
n_samples=1000000


f = h5py.File(data_file, 'r')
print(f.keys())
dat = pd.read_hdf(data_file, key="discretized", stop=None)


pt_bins,eta_bins,phi_bins=LoadBins()
print('##### GENERATED JETS ######')
print('discrete')
jets,ptj,mj=LoadSGenamples(data_file,pt_bins,eta_bins,phi_bins,n_samples)
print('cont')
print(jets)
print(np.shape(jets))
print(np.shape(mj))
print(np.shape(ptj))

SaveSamples(jets,mj,ptj,outpult_path)

ReadHDF(outpult_path)

'''
print('##### TRUE TEST JETS ######')
print('disc')
discrete_truedata_filename='/Users/humbertosmac/Documents/work/Transformers/Optimal_Classifier/test_data/discretized/TTBar_test___10M_TTBar.h5'
jets_true,ptj_true,mj_true=LoadTrue(discrete_truedata_filename,n_samples,pt_bins,eta_bins,phi_bins)
print('cont')
print(jets_true)
'''
