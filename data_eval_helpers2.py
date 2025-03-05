import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy import stats
plt.rcParams['text.usetex'] = True

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
    #print('mj :',np.min(mj),np.max(mj))
    #print('mj :',np.nanmin(mj),np.nanmax(mj))
    
    continues_jets = np.stack((pt_con, eta_con, phi_con), -1)

    return continues_jets, ptj, mj

def comp_topk_plots(mj_data,jets_data,path_to_plots,data_name,label):
    mj_true=mj_data[-2]
    mul_true=np.sum(jets_data[-2][:, :, 0] != 0, axis=1)
    mj_disc_true=mj_data[-1]
    mul_disc_true=np.sum(jets_data[-1][:, :, 0] != 0, axis=1)
    n=len(mj_data)
    wd_list=np.empty(shape=(len(mj_data)-1,4))
    if data_name=='HToGG' or data_name=='ZToNuNu':
        colors = plt.cm.rainbow_r(np.linspace(0,1,n-3))
    else:
        colors = plt.cm.rainbow_r(np.linspace(0,1,n-2))
    bins = np.linspace(0, 700, 200)
    fig,ax=plt.subplots(2,1,figsize=(12,10))
    ax[0].set_xlabel('')
    for i in range(n):
        mj=mj_data[i]
        mj=(mj[~np.isnan(mj)])
        jets=jets_data[i]
        mul=np.sum(jets[:, :, 0] != 0, axis=1)
        up_lim=min(max(mul_true),max(mul))+0.5
        low_lim=-0.5
        n_bins=int(up_lim-low_lim+1)
        mul_bins=np.linspace(low_lim,up_lim,n_bins)
        if label[i]=='JetClass training data':
            ax[0].hist(mj,bins=bins,color='black',histtype='step',density=True,alpha=0.65,log=True)
            ax[1].hist(mul,bins=mul_bins,color='black',histtype='step',density=True,alpha=0.65,label=label[i])
        elif label[i]=='JetClass':
            ax[0].hist(mj,bins=bins,color='black',linestyle='dashed',histtype='step',density=True,alpha=0.65,log=True)
            ax[1].hist(mul,bins=mul_bins,color='black',linestyle='dashed',histtype='step',density=True,alpha=0.65,label=label[i])
        elif label[i]=='k=5000, 10M':
            wd_mj_orig,ks_mj_orig=test_metrics(mj_true,mj)
            wd_mul_orig,ks_mul_orig=test_metrics(mul_true,mul)
            wd_mj_disc,ks_mj_disc=test_metrics(mj_disc_true,mj)
            wd_mul_disc,ks_mul_disc=test_metrics(mul_disc_true,mul)
            wd_list[i]=[wd_mj_orig,wd_mj_disc,wd_mul_orig,wd_mul_disc]
            #label[i]+=', WS $m_j$:orig. '+str(round(wd_mj_orig,3))+', mul: '+str(round(wd_mul,3))
            ax[0].hist(mj,bins=bins,color='black',linestyle='dotted',histtype='step',density=True,alpha=0.65,log=True,label=f'{label[i]}, WS $m_j$: orig. {str(round(wd_mj_orig,3))}, disc. {str(round(wd_mj_disc,3))}')
            ax[1].hist(mul,bins=mul_bins,color='black',linestyle='dotted',histtype='step',density=True,alpha=0.65,label=f'{label[i]}, WS $m_j$: orig. {str(round(wd_mul_orig,3))}, disc. {str(round(wd_mul_disc,3))}')
        else:
            wd_mj_orig,ks_mj_orig=test_metrics(mj_true,mj)
            wd_mul_orig,ks_mul_orig=test_metrics(mul_true,mul)
            wd_mj_disc,ks_mj_disc=test_metrics(mj_disc_true,mj)
            wd_mul_disc,ks_mul_disc=test_metrics(mul_disc_true,mul)
            wd_list[i]=[wd_mj_orig,wd_mj_disc,wd_mul_orig,wd_mul_disc]
            ax[0].hist(mj,bins=bins,color=colors[i],histtype='step',density=True,alpha=0.65,log=True,label=f'{label[i]}, WS $m_j$: orig. {str(round(wd_mj_orig,3))}, disc. {str(round(wd_mj_disc,3))}')
            ax[1].hist(mul,bins=mul_bins,color=colors[i],histtype='step',density=True,alpha=0.65,label=f'{label[i]}, WS $m_j$: orig. {str(round(wd_mul_orig,3))}, disc. {str(round(wd_mul_disc,3))}')
    ax[0].grid()
    ax[1].grid()
    ax[0].legend()
    ax[1].legend()
    ax[0].set_xlabel('$m_j$')
    ax[1].set_xlabel('Multiplicity')
    fig.suptitle(f'$m_j$-distribution for {data_name} with differnt $k$ values')
    plt.savefig(path_to_plots+'/hist_mj_topk_'+data_name+'.png')
    plt.close()
    print(wd_list)
    wd_min=np.argmin(wd_list,axis=0)
    return wd_min[0], wd_min[1]


def Make_Plots(jets,pt_bins,eta_bins,phi_bins,mj,jets_true,mj_true,path_to_plots):
    
    pt_true=(jets_true[:,:,0]).flatten()
    pt_samp=(jets[:,:,0]).flatten()
    pt_true_zero=pt_true!=0
    pt_samp_zero=pt_samp!=0
    pt_true=pt_true[pt_true_zero]
    pt_samp=pt_samp[pt_samp_zero]
    
    eta_true=jets_true[:,:,1].flatten()[pt_true_zero]
    eta_samp=jets[:,:,1].flatten()[pt_samp_zero]
    
    phi_true=jets_true[:,:,2].flatten()[pt_true_zero]
    phi_samp=jets[:,:,2].flatten()[pt_samp_zero]
    
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
    mj_true=(mj_true[~np.isnan(mj_true)])
    mj_samp=(mj[~np.isnan(mj)])
    
    #qqplots
    qq_plot(pt_true,pt_samp,'pt',path_to_plots,pt_bins,'step')
    qq_plot(eta_true,eta_samp,'eta',path_to_plots,eta_bins,'step')
    qq_plot(phi_true,phi_samp,'phi',path_to_plots,phi_bins,'step')
    qq_plot(mul_true,mul_samp,'multiplicity',path_to_plots,mul_bins,'step')
    qq_plot(mj_true,mj_samp,'mj',path_to_plots,mj_bins,'step')

    return mul_samp, mul_true, pt_samp, pt_true

def qq_plot(data_true,data_samp,data_name,path_to_plots,bins,htype):
    xlabel=data_name

    wd,ks=test_metrics(data_true,data_samp)
    plot_text='Wasserstein distance:\n   '+str(round(wd,5))+'\nKS test:\n   statistic '+str(round(ks[0],5))+', p-value '+str(round(ks[1],5))
    
    if len(data_true)!=len(data_samp):
        len_plot=min(len(data_true),len(data_samp))
        data_true_sorted=np.sort(np.random.choice(data_true,len_plot,replace=False))
        data_samp_sorted=np.sort(np.random.choice(data_samp,len_plot,replace=False))
    else:
        data_true_sorted=np.sort(data_true)
        data_samp_sorted=np.sort(data_samp)

    if data_name=='pt':
        data_true=np.log(data_true)
        data_samp=np.log(data_samp)
        xlabel='$\log (p_T)$'

    fig,(ax0,ax1,ax2)=plt.subplots(1,3,figsize=(12,5),gridspec_kw=dict(width_ratios=[0.4,0.4,0.2]))
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

    ax1.plot([0,np.nanmin(data_true_sorted),np.nanmax(data_true_sorted)],[0,np.nanmin(data_true_sorted),np.nanmax(data_true_sorted)],color='black',label='diagonal')
    ax1.scatter(data_true_sorted,data_samp_sorted,marker='.',color='blueviolet',alpha=0.65,s=3)   #zorder=2.5,
    ax1.set_xlabel('true data')
    ax1.set_ylabel('sampled data')
    ax1.legend()
    ax1.grid()
    ax2.axis('off')
    ax2.text(1.05,0.75,plot_text,transform=ax1.transAxes,fontsize=12,verticalalignment='center',bbox=dict(boxstyle='round', facecolor='white', alpha=0))
    fig.suptitle(data_name)
    plt.savefig(path_to_plots+'/hist_qqplot_wd_ks_'+data_name+'.png')
    plt.close()


def LoadTrue(discrete_truedata_filename,n_samples,pt_bins,eta_bins,phi_bins):


    tmp = pd.read_hdf(discrete_truedata_filename, key="discretized", stop=None)
    #print(tmp.shape) 
    #tmp=tmp.sample(n_samples)
    tmp = tmp.to_numpy()[:, :600].reshape(len(tmp), -1, 3)

    tmp=tmp[:,:,:]

    mask = tmp[:, :, 0] == -1
    jets_true,ptj_true,mj_true = make_continues(tmp, mask,pt_bins,eta_bins,phi_bins, noise=False)

    return jets_true,ptj_true,mj_true



def LoadSGenamples(filename,pt_bins,eta_bins,phi_bins,n_samples):

    tmp = pd.read_hdf(filename, key="discretized", stop=None)
    #print(tmp.shape)
    #tmp=tmp.sample(frac=1)
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
