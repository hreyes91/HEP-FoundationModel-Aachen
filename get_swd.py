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

get_dist=True
calc_swd=False
check=False

data=['ZToNuNu','TTBar','HToGG','WToQQ','ZToQQ']  #

pt_bins=np.load('/net/data_ttk/hreyes/OneBin/preprocessing_bins/pt_bins_1Mfromeach_403030.npy')
eta_bins=np.load('/net/data_ttk/hreyes/OneBin/preprocessing_bins/eta_bins_1Mfromeach_403030.npy')
phi_bins=np.load('/net/data_ttk/hreyes/OneBin/preprocessing_bins/phi_bins_1Mfromeach_403030.npy')

iter_swd=100
n_slices=10

def get_samp_data(name,noise=False):
    n_samples_samples=200000
    filename_samp=f'/net/data_ttk/koller/model_data/model_data_{name}_10M_train_ev_30_ep/samples_test.h5'
    jets_samp,ptj_samp,mj_samp=deh.LoadSGenamples(filename_samp,pt_bins,eta_bins,phi_bins,n_samples_samples,noise=noise)


    max_const=30    #min(np.shape(jets_true)[1],np.shape(jets_samp)[1])
    jets_samp_new=jets_samp[:,:max_const,:]
    shape_2=np.shape(jets_samp_new)
    jets_samp_par=np.reshape(jets_samp_new,(-1,shape_2[1]*shape_2[2]))
    return jets_samp_par

def get_true_data(name,true_cont=False,noise=False):
    n_samples_true=200000
    if true_cont==False:
        if name=='ZToNuNu':
            filename_disc_truedata='/net/data_ttk/koller/JetClass/discretized/ZJetsToNuNu_train___1Mfromeach_403030.h5'
        else:
            filename_disc_truedata=f'/net/data_ttk/koller/JetClass/discretized/{name}_train___1Mfromeach_403030.h5'
        jets_true,ptj_true,mj_true=deh.LoadTrue(filename_disc_truedata,n_samples_true,pt_bins,eta_bins,phi_bins,key="discretized",noise=noise)
    else:
        if name=='ZToNuNu':
            filename='/net/data_ttk/hreyes/JetClass/JetClass_pt_part/ZJetsToNuNu_train.h5'
        else:
            filename=f'/net/data_ttk/hreyes/JetClass/JetClass_pt_part/{name}_train.h5'
        jets_true,ptj_true,mj_true=deh.LoadJetClass(filename,nJets=10000000)    
    max_const=30    #min(np.shape(jets_true)[1],np.shape(jets_samp)[1])
    jets_true_new=jets_true[:,:max_const,:]
    shape_1=np.shape(jets_true_new)
    jets_true_par=np.reshape(jets_true_new,(-1,shape_1[1]*shape_1[2]))
    return jets_true_par


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

def get_swd_dist(name,iter_calc=1000,iter_swd=100,max_const=30,n_slices=10,noise=False):
    n_samples_true=200000
    if name=='ZToNuNu':
        filename='/net/data_ttk/hreyes/JetClass/JetClass_pt_part/ZJetsToNuNu_train.h5'
        filename_disc_truedata='/net/data_ttk/koller/JetClass/discretized/ZJetsToNuNu_train___1Mfromeach_403030.h5'
    else:
        filename=f'/net/data_ttk/hreyes/JetClass/JetClass_pt_part/{name}_train.h5'
        filename_disc_truedata=f'/net/data_ttk/koller/JetClass/discretized/{name}_train___1Mfromeach_403030.h5'
    jets_disc,ptj_disc,mj_disc=deh.LoadTrue(filename_disc_truedata,n_samples_true,pt_bins,eta_bins,phi_bins,key="discretized",sample=False,noise=noise)
    jets_cont,ptj_cont,mj_cont=deh.LoadJetClass(filename,nJets=10000000)    

    jets=[jets_cont,jets_disc]
    o=0
    for jets_true in jets:
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
        plt.fill_betweenx([0,1.1],t_68,t_95,color='powderblue',alpha=0.45,label=f'68\%: t={round(t_68,4)}')
        plt.fill_betweenx([0,1.1],t_95,t_99,color='powderblue',alpha=0.75,label=f'95\%: t={round(t_95,4)}')
        plt.fill_betweenx([0,1.1],t_99,max_x,color='powderblue',label=f'99\%: t={round(t_99,4)}')
        plt.legend(loc='upper right')
        plt.yscale('log')
        plt.xlim((min_x,max_x))
        plt.ylim((0,1.1))
        plt.xlabel(f'$t_{{SWD}}$')
        plt.title(f'CDF for {name} particle features, 30 constituents')
        #plt.savefig(f'plots_swd/swd_dist_{name}.png')
        if o==0:
            plt.savefig(f'plots_swd/swd_dist_{name}_jetclass.svg')
            o=1
        else:
            if noise==False:
                plt.savefig(f'plots_swd/swd_dist_{name}.svg')
            else:
                plt.savefig(f'plots_swd/swd_dist_{name}_w_noise.svg')
    return

def cross_check(name1,name2):
    data_true=get_true_data(name1)
    data_samp=get_samp_data(name2)

    swd=get_swd(data_true,data_samp,iter_swd,n_slices)
    swd_list=swd['metric_lists']
    swd_means=swd['metric_means']
    swd_stds=swd['metric_stds']

    return

if get_dist==True:
    for v in data:
        get_swd_dist(v,noise=True)

if calc_swd==True:
    noise=True
    true_cont=False
    n=len(data)

    colors=plt.cm.rainbow(np.linspace(0,1,n))
    fig,ax=plt.subplots(n,1,figsize=(12,2*n),sharex=True)

    for i in range(n):
        name=data[i]
        print(name)

        #get swd
        '''path=f'/net/data_ttk/koller/model_data/model_data_{name}_10M_train_ev_30_ep/swd_{name}_{iter_swd}_iter_{n_slices}_slices_.npy'
        check_file=os.path.isfile(path)
        if check_file:
            swd=np.load(path,allow_pickle='True',)
            print(type(swd))
            print(np.shape(swd))
            swd_list=swd[0]
            swd_means=swd[1]
            swd_stds=swd[2]
        else:'''

        data_true=get_true_data(name,true_cont=true_cont,noise=noise)
        data_samp=get_samp_data(name,noise=noise)
        swd=get_swd(data_true,data_samp,iter_swd,n_slices)
        swd_list=swd['metric_lists']
        swd_means=swd['metric_means']
        swd_stds=swd['metric_stds']

        #swd=get_swd(name,iter_swd,n_slices)

        iter_mean=np.mean(swd_means)
        iter_std=np.std(swd_means)
        mean_std=np.sqrt(np.sum([swd['metric_stds'][d]**2 for d in range(len(swd['metric_stds']))]))/len(swd['metric_stds'])

        #plot swd
        ax[i].scatter(0,0,color='white',alpha=0,label=name)
        for j in range(iter_swd):
            x=np.ones(n_slices)*(j+1)
            y=swd_list[j][:]
            ax[i].scatter(x,y,color='grey',alpha=0.4)
        ax[i].scatter(np.arange(1,iter_swd+1),swd_means,color=colors[i],label='sliced Wasserstein distance')
        ax[i].errorbar(np.arange(1,iter_swd+1),swd_means,yerr=swd_stds,color=colors[i],fmt='.')

        #plot overall mean with error
        ax[i].hlines(iter_mean,0,iter_swd+1,color=colors[i],label=f'iterations mean: {round(iter_mean,5)}$\pm${round(iter_std,3)}')
        ax[i].fill_between(np.arange(0,iter_swd+2),iter_mean-iter_std,iter_mean+iter_std,color=colors[i],alpha=0.4)
        ax[i].set_ylabel(f'swd')
        #ax[i].set_yscale('log')
        ax[i].set_xlim(0,iter_swd+1)
        ax[i].legend(loc='upper right')

    ax[-1].set_xlabel(f'iterations')
    fig.subplots_adjust(hspace=0)
    fig.suptitle(f'sliced Wasserstein distances',y=0.93,fontsize='xx-large')
    if noise==False and true_cont==False:
        plt.savefig('plots_swd/swd_seed_sl_0_30_const.svg')
    elif noise==True and true_cont==False:
        plt.savefig('plots_swd/swd_seed_sl_0_30_const_w_noise.svg')
    elif noise==False and true_cont==True:
        plt.savefig('plots_swd/swd_seed_sl_0_30_const_jetclass.svg')
    else:
        plt.savefig('plots_swd/swd_seed_sl_0_30_const_w_noise_jetclass.svg')

    plt.close()


if check==True:
    cross_check(data[0],data[1])