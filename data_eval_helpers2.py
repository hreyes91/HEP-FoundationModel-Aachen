import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy import stats
#plt.rcParams['text.usetex'] = True

def make_continues(jets, mask,pt_bins,eta_bins,phi_bins, noise=False):


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


    pt_con = np.exp(pt_con)
    
    
    pt_con[mask] = 0.0
    eta_con[mask] = 0.0
    phi_con[mask] = 0.0
    
    pxs = np.cos(phi_con) * pt_con
    pys = np.sin(phi_con) * pt_con
    pzs = np.sinh(eta_con) * pt_con
    es = (pxs ** 2 + pys ** 2 + pzs ** 2) ** (1. / 2)

    pxj = np.sum(pxs, -1)
    pyj = np.sum(pys, -1)
    pzj = np.sum(pzs, -1)
    ej = np.sum(es, -1)
    
    ptj = np.sqrt(pxj**2 + pyj**2)
    mj = (ej ** 2 - pxj ** 2 - pyj ** 2 - pzj ** 2) ** (1. / 2)
    print('mj :',np.min(mj),np.max(mj))
    print('mj :',np.nanmin(mj),np.nanmax(mj))
    
    continues_jets = np.stack((pt_con, eta_con, phi_con), -1)

    return continues_jets, ptj, mj

def Make_Plots(jets,pt_bins,eta_bins,phi_bins,mj,jets_true,ptj_true,mj_true,path_to_plots,n_events,n_epochs,n_samples,data_class):
    
    pt_true=(jets_true[:,:,0]).flatten()
    pt_samp=(jets[:,:,0]).flatten()
    
    eta_true=jets_true[:,:,1].flatten()
    eta_samp=jets[:,:,1].flatten()
    
    phi_true=jets_true[:,:,2].flatten()
    phi_samp=jets[:,:,2].flatten()
    
    mul_true=np.sum(jets_true[:, :, 0] != 0, axis=1)
    mul_samp=np.sum(jets[:, :, 0] != 0, axis=1)
    up_lim=min(max(mul_true),max(mul_samp))+0.5
    low_lim=-0.5
    n_bins=int(up_lim-low_lim+1)

    mul_bins=np.linspace(low_lim,up_lim,n_bins)
    mul_true=np.clip(mul_true,0,int(up_lim),dtype=int)
    
    mj_bins = np.linspace(0, 700, 200)
    #mj_true=np.clip(mj_true, mj_bins[0], mj_bins[-1])
    #mj_samp=np.clip(mj, mj_bins[0], mj_bins[-1])
    mj_samp=np.ma.masked_invalid(mj)
    mj_true=np.ma.masked_invalid(mj_true)
    
    #qqplots
    qq_plot(pt_true,pt_samp,'pt',path_to_plots,pt_bins,'step')
    qq_plot(eta_true,eta_samp,'eta',path_to_plots,eta_bins,'step')
    qq_plot(phi_true,phi_samp,'phi',path_to_plots,phi_bins,'step')
    qq_plot(mul_true,mul_samp,'multiplicity',path_to_plots,mul_bins,'step')
    qq_plot(mj_true,mj_samp,'mj',path_to_plots,mj_bins,'step')

    wd_mul,ks_mul=test_metrics(mul_true,mul_samp)
    print("wasserstein distance multiplicity:",wd_mul)
    print("ks test multiplicity:",ks_mul)

    wd_pt,ks_pt=test_metrics(pt_true,pt_samp)
    print("wasserstein distance pt:",wd_pt)
    print("ks test pt:",ks_pt)

    wd_eta,ks_eta=test_metrics(eta_true,eta_samp)
    print("wasserstein distance eta:",wd_eta)
    print("ks test eta:",ks_eta)

    wd_phi,ks_phi=test_metrics(phi_true,phi_samp)
    print("wasserstein distance phi:",wd_phi)
    print("ks test phi:",ks_phi)

    wd_mj,ks_mj=test_metrics(mj_true,mj_samp)
    print("wasserstein distance mj:",wd_mj)
    print("ks test mj:",ks_mj)

    return mul_samp, mul_true, pt_samp, pt_true

def qq_plot(data_true,data_samp,data_name,path_to_plots,bins,htype):
    xlabel=data_name

    #wd,ks=test_metrics(data_true,data_samp)
    #plot_text='Wasserstein distance:\n   '+str(round(wd,5))+'\nKS test:\n   statistic '+str(round(ks[0],5))+', p-value '+str(round(ks[1],5))
    
    if len(data_true)!=len(data_samp):
        len_plot=min(len(data_true),len(data_samp))
        data_true_sorted=np.sort(np.random.choice(data_true,len_plot,replace=False))
        data_samp_sorted=np.sort(np.random.choice(data_samp,len_plot,replace=False))
    else:
        data_true_sorted=np.sort(data_true)
        data_samp_sorted=np.sort(data_samp)


    xmin=np.nanmin(data_true)-0.1*np.absolute(np.nanmin(data_true))
    xmax=1.1*np.nanmax(data_true)
    ymin=np.nanmin(data_samp)-0.1*np.absolute(np.nanmin(data_samp))
    ymax=1.1*np.nanmax(data_samp)
    pos_left=0.1
    pos_right=0.8
    xtext=(xmax-xmin)/(pos_right-pos_left)*xmax
    ytext=0.5*(ymin+ymax)

    if data_name=='pt':
        data_true=np.log(data_true)
        data_samp=np.log(data_samp)
        xlabel='$\log (p_T)$'

    fig,(ax0,ax1)=plt.subplots(1,2,figsize=(10,5))
    '''hist_true,_=np.histogram(data_true,bins=bins,density=True)
    hist_samp,_=np.histogram(data_samp,bins=bins,density=True)
    hist=hist_true-hist_samp'''
    #ax2=plt.subplot(212)
    #ax2.hist(hist,bins=bins,density=True,color='seagreen')

    #ax0=plt.subplot(221)
    if data_name=='mj':
        ax0.hist(data_true,bins=bins,color='dodgerblue',label='true',histtype=htype,density=True,log=True)
        ax0.hist(data_samp,bins=bins,color='red',label='sampled',histtype=htype,density=True,log=True)
    else:
        ax0.hist(data_true,bins=bins,color='dodgerblue',label='true',histtype=htype,density=True)
        ax0.hist(data_samp,bins=bins,color='red',label='sampled',histtype=htype,density=True)
    ax0.set_xlabel(xlabel)
    '''if data_name=='mj':
        ax0.set_xlim(0,200)
    else:
        ax0.set_xlim(1.01*min(bins),1.01*max(bins))'''
    ax0.set_xlim(1.01*min(bins),1.01*max(bins))
    if data_name=='multiplicity':
        ax0.set_xticks([x+0.5 for x in bins[::10]])
    ax0.grid()
    ax0.legend()
    #ax0.set_title('Normalized distribution')

    #ax1=plt.subplot(222)
    ax1.plot([0,np.nanmin(data_true_sorted),np.nanmax(data_true_sorted)],[0,np.nanmin(data_true_sorted),np.nanmax(data_true_sorted)],color='black',label='diagonal')
    ax1.scatter(data_true_sorted,data_samp_sorted,marker='.',color='blueviolet',alpha=0.65,s=3)   #zorder=2.5,
    ax1.set_xlabel('true data')
    ax1.set_ylabel('sampled data')
    #ax1.set_ylim(ymin,ymax)
    #ax1.set_title('Quantile-quantile plot')
    ax1.legend()
    ax1.grid()
    #plt.text(xtext,ytext,plot_text,dict(size=10))
    fig.suptitle(data_name)
    plt.savefig(path_to_plots+'/hist_qqplot_'+data_name+'.png')
    plt.close()


def LoadTrue(discrete_truedata_filename,n_samples,pt_bins,eta_bins,phi_bins):


    tmp = pd.read_hdf(discrete_truedata_filename, key="discretized", stop=None)
    #print(tmp.shape) 
    tmp=tmp.sample(n_samples)
    tmp = tmp.to_numpy()[:, :600].reshape(len(tmp), -1, 3)

    tmp=tmp[:,:,:]

    mask = tmp[:, :, 0] == -1
    jets_true,ptj_true,mj_true = make_continues(tmp, mask,pt_bins,eta_bins,phi_bins, noise=False)

    return jets_true,ptj_true,mj_true



def LoadSGenamples(filename,pt_bins,eta_bins,phi_bins,n_samples):

    tmp = pd.read_hdf(filename, key="discretized", stop=None)
    #print(tmp.shape)
    tmp=tmp.sample(frac=1)
    tmp = tmp.to_numpy()[:, :600].reshape(len(tmp), -1, 3)

    mask = tmp[:, :, 0] == -1
    jets,ptj,mj = make_continues(tmp, mask,pt_bins,eta_bins,phi_bins, noise=False)

    return jets,ptj,mj

def GetHighLevel(jets):

    pt=jets[:,:,0]
    
    eta=jets[:,:,1]
    phi=jets[:,:,2]
    mul=np.sum(jets[:, :, 0] != 0, axis=1)
    return pt, eta,phi,mul

def test_metrics(data_true,data_samp):
    #print(np.argwhere(np.isnan(data_true)))
    #print(np.argwhere(np.isnan(data_samp)))

    data_samp=np.ma.masked_invalid(data_samp)
    data_true=np.ma.masked_invalid(data_true)
    #print(np.argwhere(np.isnan(data_samp)))

    max=min(np.max(data_samp),np.max(data_true))
    data_samp=np.delete(data_samp,np.where(data_samp>=max))
    data_true=np.delete(data_true,np.where(data_true>=max))


    #plt.plot(np.sort(data_true),label='true')   #,marker='.',s=5)
    #plt.plot(np.sort(data_samp),label='samp')   #,marker='.',s=5)
    #plt.plot((np.sort(data_true)-np.sort(data_samp)))
    #plt.savefig('plots_TTBar_10M_events_30_epochs_200k_samples/scatter_mj_data_diff')
    w_distance=stats.wasserstein_distance(data_true,data_samp)
    ks_res=stats.ks_2samp(data_true,data_samp,alternative='two-sided',nan_policy='omit')

    return w_distance,ks_res


def read_file(file_name):

    f = open(file_name, "r")
    lines = f.readlines()

    return lines


def extract_value(var,lines):

    for line in lines:
        if 'lr_' in line:
            continue
        if var in line:
            line=line.replace(' ','')
            line=line.replace('\n','')

            value=line.split(var)[-1]
    
    return value
