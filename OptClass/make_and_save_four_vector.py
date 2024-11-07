import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy import stats
from data_eval_helpers import  LoadTrue


def LoadSGenamples(filename,pt_bins,eta_bins,phi_bins,n_samples):

    tmp = pd.read_hdf(filename, key="discretized", stop=None)
    #tmp=tmp.sample(n_samples)

    tmp = tmp.to_numpy()[:, :600].reshape(len(tmp), -1, 3)
    print(tmp)

    mask = tmp[:, :, 0] == -1
    continues_jets_4v = make_fourvectors(tmp, mask,pt_bins,eta_bins,phi_bins, noise=False)

    return continues_jets_4v

def make_fourvectors(jets, mask,pt_bins,eta_bins,phi_bins, noise=True):


    pt_disc = jets[:, :, 0]
    eta_disc = jets[:, :, 1]
    phi_disc = jets[:, :, 2]

    if noise:
        pt_con = (pt_disc - np.random.uniform(0.0, 1.0, size=pt_disc.shape)) * (
            pt_bins[1] - pt_bins[0]
        ) + pt_bins[0]
        eta_con = (eta_disc - np.random.uniform(0.0, 1.0, size=eta_disc.shape)) * (
            eta_bins[1] - eta_bins[0]
        ) + eta_bins[0]
        phi_con = (phi_disc - np.random.uniform(0.0, 1.0, size=phi_disc.shape)) * (
            phi_bins[1] - phi_bins[0]
        ) + phi_bins[0]
    else:
        pt_con = (pt_disc - 0.5) * (pt_bins[1] - pt_bins[0]) + pt_bins[0]
        eta_con = (eta_disc - 0.5) * (eta_bins[1] - eta_bins[0]) + eta_bins[0]
        phi_con = (phi_disc - 0.5) * (phi_bins[1] - phi_bins[0]) + phi_bins[0]


    print(pt_con)
    print(np.shape(pt_con))
    pt_con = np.exp(pt_con)
    print(pt_con)
    print(np.shape(pt_con))
    pt_jet=np.sum(pt_con,axis=-1)
    
    print(pt_jet)
    print(np.shape(pt_jet))
    
    
   
    
    
    pt_con[mask] = 0.0
    eta_con[mask] = 0.0
    phi_con[mask] = 0.0
    print(pt_con)
    print(np.shape(pt_con))
    print('pxs')
    pxs = np.cos(phi_con) * pt_con
    print(pxs)
    print(np.shape(pxs))
    print('phi_con')
    print(phi_con)
    print(np.shape(phi_con))
    pys = np.sin(phi_con) * pt_con
    pzs = np.sinh(eta_con) * pt_con
    es = (pxs ** 2 + pys ** 2 + pzs ** 2) ** (1. / 2)

    print('pt jet')
    
    
    pxj = np.sum(pxs, -1)
    pyj = np.sum(pys, -1)
    pzj = np.sum(pzs, -1)
    ej = np.sum(es, -1)
    pt_jet=np.sqrt(pxj**2 + pyj**2)
    print(pt_jet)
    print(np.shape(pt_jet))
    
    exit()

    continues_jets_4v = np.stack((pxs, pys, pzs,es), -1)

    return continues_jets_4v

def LoadBins():
    bins_path_prefix='/Users/humbertosmac/Documents/work/Transformers/Optimal_Classifier/LLRModels/preprocessing_bins/'
    pt_bins = np.load(bins_path_prefix+'pt_bins_10M_TTBar.npy')
    eta_bins = np.load(bins_path_prefix+'eta_bins_10M_TTBar.npy')
    phi_bins = np.load(bins_path_prefix+'phi_bins_10M_TTBar.npy')
    print('pt_bins')

    return pt_bins,eta_bins,phi_bins


def SaveSamples(continues_jets,outpult_path):
    hf = h5py.File(outpult_path, 'w')
    hf.create_dataset('vector4', data=continues_jets)
    
    hf.close()
    return


def ReadHDF(outpult_path):


    #tmp = pd.read_hdf(outpult_path, key="raw", stop=None)
    tmp = h5py.File(outpult_path, 'r')
    print(np.array(tmp.get('raw')))
    return

data_file='/Users/humbertosmac/Documents/work/Transformers/Optimal_Classifier/TheSamples/FirstTime_topvsqcd_100const/top/discrete/samples_samples_nsamples200000_trunc_5000.h5'

outpult_path='/Users/humbertosmac/Documents/work/Transformers/Optimal_Classifier/TheSamples/FirstTime_topvsqcd_100const/top/vector4/just_testingtrain_nsamples200000_trunc_5000.h5'
n_samples=200000


f = h5py.File(data_file, 'r')
print(f.keys())
dat = pd.read_hdf(data_file, key="discretized", stop=None)


pt_bins,eta_bins,phi_bins=LoadBins()
print('##### GENERATED JETS ######')
print('discrete')
continues_jets_4v=LoadSGenamples(data_file,pt_bins,eta_bins,phi_bins,n_samples)
print('cont')
print(continues_jets_4v[:3, :, :])
print(np.shape(continues_jets_4v))

SaveSamples(continues_jets_4v,outpult_path)

ReadHDF(outpult_path)

'''
print('##### TRUE TEST JETS ######')
print('disc')
discrete_truedata_filename='/Users/humbertosmac/Documents/work/Transformers/Optimal_Classifier/test_data/discretized/TTBar_test___10M_TTBar.h5'
jets_true,ptj_true,mj_true=LoadTrue(discrete_truedata_filename,n_samples,pt_bins,eta_bins,phi_bins)
print('cont')
print(jets_true)
'''
