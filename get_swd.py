#import numpy as np
import os
import h5py
from scipy import stats
import matplotlib.pyplot as plt


import numpy as np
import tensorflow as tf
#import tensorflow_probability as tfp
#import GenerativeModelsMetrics as GMetrics


import data_eval_helpers2 as deh
import GMetrics
from GMetrics.utils import reset_random_seeds
from GMetrics.utils import conditional_print
from GMetrics.utils import conditional_tf_print
from GMetrics.utils import generate_and_clean_data
from GMetrics.utils import NumpyDistribution
from GMetrics.base import TwoSampleTestInputs
from GMetrics.base import TwoSampleTestSlicedBase
from GMetrics.base import TwoSampleTestResult
from GMetrics.base import TwoSampleTestResults
from GMetrics.swd import SWDMetric

from typing import Tuple, Union, Optional, Type, Dict, Any, List
from GMetrics.utils import DTypeType, IntTensor, FloatTensor, BoolTypeTF, BoolTypeNP, IntType, DataTypeTF, DataTypeNP, DataType, DistTypeTF, DistTypeNP, DistType, DataDistTypeNP, DataDistTypeTF, DataDistType, BoolType

data=['ZToNuNu','TTBar','HToGG','WToQQ','ZToQQ']  #

pt_bins=np.load('/net/data_ttk/hreyes/OneBin/preprocessing_bins/pt_bins_1Mfromeach_403030.npy')
eta_bins=np.load('/net/data_ttk/hreyes/OneBin/preprocessing_bins/eta_bins_1Mfromeach_403030.npy')
phi_bins=np.load('/net/data_ttk/hreyes/OneBin/preprocessing_bins/phi_bins_1Mfromeach_403030.npy')

iter_swd=100
n_slices=10

def get_data(name):
    n_samples_samples=200000
    filename_samp=f'/net/data_ttk/koller/model_data/model_data_{name}_10M_train_ev_30_ep/samples_test.h5'
    jets_samp,ptj_samp,mj_samp=deh.LoadSGenamples(filename_samp,pt_bins,eta_bins,phi_bins,n_samples_samples)

    n_samples_true=200000
    if name=='ZToNuNu':
        filename_disc_truedata='/net/data_ttk/koller/JetClass/discretized/ZJetsToNuNu_train___1Mfromeach_403030.h5'
    else:
        filename_disc_truedata=f'/net/data_ttk/koller/JetClass/discretized/{name}_train___1Mfromeach_403030.h5'
    jets_true,ptj_true,mj_true=deh.LoadTrue(filename_disc_truedata,n_samples_true,pt_bins,eta_bins,phi_bins,key="discretized")
    max_const=30    #min(np.shape(jets_true)[1],np.shape(jets_samp)[1])
    jets_true_new=jets_true[:,:max_const,:]
    jets_samp_new=jets_samp[:,:max_const,:]
    shape_1=np.shape(jets_true_new)
    shape_2=np.shape(jets_samp_new)
    jets_true_par=np.reshape(jets_true_new,(-1,shape_1[1]*shape_1[2]))#,order='F')
    jets_samp_par=np.reshape(jets_samp_new,(-1,shape_2[1]*shape_2[2]))#,order='F')
    return jets_true_par,jets_samp_par



#get sliced wasserstein distance
def get_swd(data_true,data_samp,iter_swd=100,n_slices=10):
    data_input_low_level=TwoSampleTestInputs(dist_1_input = data_true,
                                                    dist_2_input = data_samp,
                                                    niter = iter_swd,
                                                    dtype_input = tf.float64,
                                                    #seed_input = 15,
                                                    use_tf = True,
                                                    verbose = True)
    swd_metric=SWDMetric(data_input = data_input_low_level,
                                            nslices=n_slices,
                                            seed_slicing=0,        #what does seed_slicing do? doesnt work without it
                                            progress_bar = True, 
                                            verbose = True)
    swd_metric.compute(max_vectorize = int(1e6))    #max_vectorize = int(1e6)
    swd_res=swd_metric.Results[0].result_value       #refer to ['metric_lists'],['metric_means'], ['metric_stds'] for single arrays
    #output: metric_lists: iter_swd * nslices values; metric_means: iter_swd means( one per slice=direction): metric_stds: std of each mean
    return swd_res

def get_swd_dist(name,iter_calc=1000,iter_swd=100,max_const=30,n_slices=10,jetclass=False):
    n_samples_true=200000
    if jetclass==True:
        if name=='ZToNuNu':
            filename='/net/data_ttk/hreyes/JetClass/JetClass_pt_part/ZJetsToNuNu_train.h5'
        else:
            filename=f'/net/data_ttk/hreyes/JetClass/JetClass_pt_part/{name}_train.h5'
        jets_true,ptj_true,mj_true=deh.LoadJetClass(filename,nJets=10000000)    
    else:
        if name=='ZToNuNu':
            filename_disc_truedata='/net/data_ttk/koller/JetClass/discretized/ZJetsToNuNu_train___1Mfromeach_403030.h5'
        else:
            filename_disc_truedata=f'/net/data_ttk/koller/JetClass/discretized/{name}_train___1Mfromeach_403030.h5'
        jets_true,ptj_true,mj_true=deh.LoadTrue(filename_disc_truedata,n_samples_true,pt_bins,eta_bins,phi_bins,key="discretized",sample=False)

    #only use first 30 constituents
    jets_true=jets_true[:,:max_const,:]
    jets_true=np.reshape(jets_true,(-1,3*max_const))

    #get shuffled version of data
    jets_random=np.random.permutation(jets_true)

    #divide data into iter_calc batches
    jets_1=np.reshape(jets_true,(iter_calc,-1,3*max_const))
    jets_2=np.reshape(jets_random,(iter_calc,-1,3*max_const))

    res_mean=np.empty((iter_calc,iter_swd))
    res_stds=np.empty((iter_calc,iter_swd))
    for k in range(iter_calc):
        swd=get_swd(jets_1[k],jets_2[k])
        res_mean[k]=swd['metric_means']
        res_stds[k]=swd['metric_stds']
    
    means=np.reshape(res_mean,(-1))
    stds=np.reshape(res_stds,(-1))

    min_x=0.95*min(means)
    max_x=1.05*max(means)
    bins=np.arange(min_x,max_x,(max_x-min_x)/100)
    plt.figure(figsize=(8,5))
    n,bins,patches=plt.hist(means,bins=bins,density=True,cumulative=True,color='black',histtype='step')
    t_68=bins[np.where(n>=0.68)[0][0]]
    t_95=bins[np.where(n>=0.95)[0][0]]
    t_99=bins[np.where(n>=0.99)[0][0]]
    print(t_68)
    plt.hlines(1,min_x,max_x,color='dimgrey',linestyle='dashed')
    plt.hlines([0.68,0.95,0.99],min_x,max_x,color='darkgray',linestyle='dotted')
    plt.vlines([t_68,t_95,t_99],0,1.1,color='dimgrey')
    plt.fill_betweenx([0,1.1],t_68,t_95,color='powderblue',alpha=0.45,label=f'68\%: t={t_68}')
    plt.fill_betweenx([0,1.1],t_95,t_99,color='powderblue',alpha=0.75,label=f'95\%: t={t_95}')
    plt.fill_betweenx([0,1.1],t_99,max_x,color='powderblue',label=f'99\%: t={t_99}')
    plt.legend(loc='upper right')
    plt.yscale('log')
    plt.xlim((min_x,max_x))
    plt.ylim((0,1.1))
    plt.xlabel(f'$t_{{SWD}}$')
    plt.title(f'CDF for {name} particle features, 30 constituents')
    #plt.savefig(f'plots_swd/swd_dist_{name}.png')
    if jetclass==True:
        plt.savefig(f'plots_swd/swd_dist_{name}_jetclass.svg')
    else:
        plt.savefig(f'plots_swd/swd_dist_{name}.svg')
    return

for v in data:
    get_swd_dist(v,jetclass=True)




