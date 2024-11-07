import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy import stats
from data_eval_helpers import make_continues,LoadSGenamples, LoadTrue



def ReadHDF(outpult_path):


    #tmp = pd.read_hdf(outpult_path, key="raw", stop=None)
    tmp = h5py.File(outpult_path, 'r')
    print(np.array(tmp.get('raw')))
    
    
    ptj=np.array(tmp.get('ptj'))
    m_jet=np.array(tmp.get('m_jet'))
    
    raw=np.array(tmp.get('raw'))
    
    return raw, ptj, m_jet
    
    

def PlotMjet(mj,outpult_path):
    path_to_plots=outpult_path
    mj_bins = np.linspace(0, 450, 100)
    plt.hist(np.clip(mj, mj_bins[0], mj_bins[-1]), bins=mj_bins,color='blue',histtype='step',density=True)
    plt.yscale('log')
    plt.xlabel('$m_{jet}$')
    plt.savefig(path_to_plots+'/plot_mj_trans_200k.png')
    plt.close()
    
    return


def PlotPTjet(ptj,outpult_path):
    path_to_plots=outpult_path
    ptj_bins = np.linspace(-20, 1500, 100)
    plt.hist(np.clip(ptj, ptj_bins[0], ptj_bins[-1]), bins=ptj_bins,color='blue',histtype='step',density=True)
    plt.yscale('log')
    plt.xlabel('$p_{jet}$')
    plt.savefig(path_to_plots+'/plot_ptj_trans_200k.png')
    plt.close()
    
    return


def Make_Plots(jets,mj,pt_bins,eta_bins,phi_bins,path_to_plots):

    plt.hist(np.log(jets[:,:,0]).flatten(), bins=pt_bins, color='blue',histtype='step',density=True)
    #plt.hist(np.log(jets_true[:,:,0]).flatten(), bins=pt_bins, color='red',histtype='step',density=True)
    plt.xlabel('$\log (p_T)$')
    
    plt.savefig(path_to_plots+'/plot_pt_trans.png')
    plt.close()
    
    plt.hist(jets[:,:,1].flatten(), bins=eta_bins, color='blue',histtype='step',density=True)
    #plt.hist(jets_true[:,:,1].flatten(), bins=eta_bins, color='red',histtype='step',density=True)
    plt.xlabel('$\Delta\eta$')
    plt.savefig(path_to_plots+'/plot_eta_trans.png')
    plt.close()
    
    plt.hist(jets[:,:,2].flatten(), bins=phi_bins, color='blue',histtype='step',density=True)
    #plt.hist(jets_true[:,:,2].flatten(), bins=phi_bins, color='red',histtype='step',density=True)
    plt.xlabel('$\Delta\phi$')
    plt.savefig(path_to_plots+'/plot_phi_trans.png')
    plt.close()
    
    mask = jets[:, :, 0] != 0
    plt.hist(np.sum(mask, axis=1), bins=np.linspace(-0.5, 100.5, 102),color='blue',histtype='step',density=True)
    #mask = jets_true[:, :, 0] != 0
    #plt.hist(np.sum(mask, axis=1), bins=np.linspace(-0.5, 100.5, 102),color='red',histtype='step',density=True)
    plt.xlabel('Multiplicity')
    plt.savefig(path_to_plots+'/plot_mul_trans.png')
    plt.close()
    

    return
    
def SeeData(mj,ptj,raw):
    
    
    print('mj')
    print(mj)
    print(np.count_nonzero(np.isnan(mj)))

    print(np.isnan(mj))
    print('ptj')
    print(ptj)
    print(np.count_nonzero(np.isnan(ptj)))
    
    mj_nan=np.isnan(mj).any(axis=0)
    print('mj_nan')
    print(mj_nan)
    return


def identify_nan_rows(mj):
    # Check where NaN values are in the array
    nan_mask = np.isnan(mj)
    print('nan mask')
    print(nan_mask)
    # Identify rows with at least one NaN value
    rows_with_nan = np.any(nan_mask, axis=-1)
    print('rows with nan')
    print(rows_with_nan)
    # Get the indices of these rows
    nan_rows_indices = np.where(rows_with_nan)[0]
    
    return nan_rows_indices

def CheckInf(mj,ptj,raw):
    
    
    print('mj')
    print(mj)
    print(np.count_nonzero(np.isinf(mj)))


    print('ptj')
    print(ptj)
    print(np.count_nonzero(np.isinf(ptj)))
    
    
    
    return


def SaveSamples(continues_jets,mj,ptj,output_file_new):
    hf = h5py.File(output_file_new, 'w')
    hf.create_dataset('raw', data=continues_jets)
    hf.create_dataset('m_jet', data=mj)
    hf.create_dataset('ptj', data=ptj)
    
    hf.close()
    return

def LoadBins():
    bins_path_prefix='/Users/humbertosmac/Documents/work/Transformers/Optimal_Classifier/LLRModels/preprocessing_bins/'
    pt_bins = np.load(bins_path_prefix+'pt_bins_40_30_30_pt_part_ZJetsToNuNu.npy')
    eta_bins = np.load(bins_path_prefix+'eta_bins_40_30_30_pt_part_ZJetsToNuNu.npy')
    phi_bins = np.load(bins_path_prefix+'phi_bins_40_30_30_pt_part_ZJetsToNuNu.npy')
    print('pt_bins')

    return pt_bins,eta_bins,phi_bins


outpult_path='/Users/humbertosmac/Dropbox/Transformers/OptimalClassifier/Data/TopVSQCD_pt_part_128const/qcd/raw/'

output_file_gen=outpult_path+'/train_nsamples1M_trunc_5000.h5'

output_file_new=outpult_path+'/train_nsamples1M_trunc_5000_nonan.h5'
output_file_true_post=outpult_path+'/train_nsamples1M_trunc_5000.h5'
output_file_true_true=outpult_path+'/train_nsamples1M_trunc_5000.h5'


n_samples=1000000


raw, ptj, mj=ReadHDF(output_file_gen)
#raw_true_post, ptj_true_post, mj_true_post=ReadHDFTruePost(output_file_true_post)
#raw_true, ptj_true, mj_true=ReadHDFTrue(output_file_true_true)



PlotMjet(mj,outpult_path)

PlotPTjet(ptj,outpult_path)
pt_bins,eta_bins,phi_bins=LoadBins()


Make_Plots(raw,mj,pt_bins,eta_bins,phi_bins,outpult_path)

SeeData(mj,ptj,raw)
CheckInf(mj,ptj,raw)
nan_indices=identify_nan_rows(mj)
print(nan_indices)
print(np.shape(nan_indices))

i=-1
print(np.nan)
for mj_i in mj:
    i=i+1
    if isinstance(mj_i, float):
        continue
    else:
        print(mj_i)
        print(i)
    
i=-1

nan_indices=[]
for item in np.isnan(mj):
    i=i+1
    if item==True:
        print(item)
        print(i)
        nan_indices.append(i)



print(nan_indices)
print(mj[nan_indices])

print(np.shape(raw))
print(raw[nan_indices])
print(np.shape(raw[nan_indices,:,:]))

new_raw= np.delete(raw, nan_indices, axis=0)
print(new_raw)
print(np.shape(new_raw))


new_mj= np.delete(mj, nan_indices, axis=0)
print(new_mj)
print(np.shape(new_mj))

new_ptj= np.delete(ptj, nan_indices, axis=0)
print(new_ptj)
print(np.shape(new_ptj))


SaveSamples(new_raw,new_mj,new_ptj,output_file_new)
