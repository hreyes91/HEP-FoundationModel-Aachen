import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
from scipy import stats
from tqdm import trange
from sklearn.metrics import roc_curve,roc_auc_score


def plot_ll_dist(results,title,name,label_num):
    label=[['hgg samples','hgg JetClass','top samples','top JetClass','wqq samples','wqq JetClass','zqq samples','zqq JetClass','qcd samples','qcd JetClass']
           ,['top model, top samples','top model, qcd samples','qcd model, qcd samples','qcd model, top samples']
           ,['zqq model, zqq samples','zqq model, qcd samples','qcd model, qcd samples','qcd model, zqq samples']
           ,['wqq model, wqq samples','wqq model, qcd samples','qcd model, qcd samples','qcd model, wqq samples']
           ,['hgg model, hgg samples','hgg model, qcd samples','qcd model, qcd samples','qcd model, hgg samples']]
    label=label[label_num]
    color=[['xkcd:light grey blue','xkcd:gunmetal','xkcd:ocean blue','xkcd:dark grey blue','xkcd:dusk','xkcd:indigo','xkcd:pine','xkcd:bottle green','xkcd:rust','xkcd:reddish']
           ,['xkcd:ocean blue','xkcd:rust','xkcd:indian red','xkcd:light navy']
           ,['xkcd:pine','xkcd:rust','xkcd:indian red','xkcd:dark green']
           ,['xkcd:dusk','xkcd:rust','xkcd:indian red','xkcd:twilight']
           ,['xkcd:light grey blue','xkcd:rust','xkcd:indian red','xkcd:slate']]
    color=color[label_num]
    htype=[['bar','step','bar','step','bar','step','bar','step','bar','step']
           ,['bar','bar','step','step']
           ,['bar','bar','step','step']
           ,['bar','bar','step','step']
           ,['bar','bar','step','step']]
    htype=htype[label_num]
    n_bins=150
    plt.figure(figsize=(10,4))
    plt.grid(which='both',color='grey',lw=0.5)
    for i in range(len(results)):
        plt.hist(results[i],bins=n_bins,histtype=htype[i],density=True,label=label[i],color=color[i],alpha=0.7)
    plt.xlim(-800,0)
    plt.xlabel('log-likelihood')
    plt.ylabel('')
    plt.legend()
    plt.title(title)
    plt.savefig('plots_llr/plot_ll_1Mevents_'+name+'.png')

def plot_llr_roc(data,label,name,num):
    color=['xkcd:ocean blue','xkcd:pine','xkcd:dusk','xkcd:light grey blue']
    plt.figure(figsize=(10,4))
    plt.hist(data[0],bins=125,range=(-45,30),color=color[num],label=label[0],alpha=0.7,density=True)
    plt.hist(data[1],bins=125,range=(-45,30),color='xkcd:rust',label=label[1],alpha=0.7,density=True)
    plt.legend()
    plt.xlim((-45,30))
    plt.grid(which='both',color='grey',lw=0.5)
    plt.xlabel('log-likelihood ratio')
    plt.ylabel('normalized distribution')
    plt.savefig('plots_llr/plot_llr_'+name+'.png')

    labels=np.append(np.zeros(len(data[1])),np.ones(len(data[0])))
    predictions=np.append(data[1],data[0])
    roc=roc_curve(labels,predictions)
    auc=roc_auc_score(labels,predictions)
    R_30_ind=np.argmin(np.abs(roc[1]-0.3))
    #print(R_30_ind,roc[1][R_30_ind-1],roc[1][R_30_ind],roc[1][R_30_ind+1])
    R_30=1/roc[0][R_30_ind]
    R_50_ind=np.argmin(np.abs(roc[1]-0.5))
    #print(R_50_ind,roc[1][R_50_ind-1],roc[1][R_50_ind],roc[1][R_50_ind+1])
    R_50=1/roc[0][R_50_ind]
    plt.figure()
    plt.scatter(roc[1],1/roc[0],color='orangered',marker='.',zorder=3,s=5)
    plt.scatter(roc[1][R_30_ind],R_30,color='black',marker='.',label='$R_{30}=$'+str(round(R_30,5)),zorder=5)
    plt.scatter(roc[1][R_50_ind],R_50,color='black',marker='.',label='$R_{50}=$'+str(round(R_50,5)),zorder=5)
    plt.hlines([R_30,R_50],[0,0],[roc[1][R_30_ind],roc[1][R_50_ind]],color='black',linestyle='dashed',linewidth=1)
    plt.vlines([roc[1][R_30_ind],roc[1][R_50_ind]],[0,0],[R_30,R_50],color='black',linestyle='dashed',linewidth=1)
    plt.scatter(0,0,color='white',label='auc: '+str(auc))
    plt.xlim((0,1))
    plt.yscale('log')
    plt.xlabel('$\epsilon_{top}$')
    plt.ylabel('$1/\epsilon_{QCD}$')
    plt.legend()
    plt.title('ROC-curve '+label[0]+'/'+label[1])

    #print('auc: ',auc)
    #plt.text(0.6,13000,'auc: '+str(auc))
    plt.grid(which='both',color='grey',lw=0.5)
    plt.savefig('plots_llr/plot_roc_'+name+'.png')


data_qcd_JetClass=np.load('/net/data_ttk/koller/model_data/model_data_ZToNuNu_10M_train_ev_30_ep/results_test.npz')['probs']

data_qcd_qcd_samples=np.load('/net/data_ttk/koller/model_data/model_data_ZToNuNu_10M_train_ev_30_ep/results_qcd_samples.npz')['probs']

data_qcd_top_samples=np.load('/net/data_ttk/koller/model_data/model_data_ZToNuNu_10M_train_ev_30_ep/results_top_samples.npz')['probs']
data_qcd_w_samples=np.load('/net/data_ttk/koller/model_data/model_data_ZToNuNu_10M_train_ev_30_ep/results_w_samples.npz')['probs']
data_qcd_z_samples=np.load('/net/data_ttk/koller/model_data/model_data_ZToNuNu_10M_train_ev_30_ep/results_z_samples.npz')['probs']
data_qcd_h_samples=np.load('/net/data_ttk/koller/model_data/model_data_ZToNuNu_10M_train_ev_30_ep/results_h_samples.npz')['probs']

data_ttbar_JetClass=np.load('/net/data_ttk/koller/model_data/model_data_TTBar_10M_train_ev_30_ep/results_jetclass.npz')['probs']
data_ttbar_top_samples=np.load('/net/data_ttk/koller/model_data/model_data_TTBar_10M_train_ev_30_ep/results_top_samples.npz')['probs']

data_ttbar_qcd_samples=np.load('/net/data_ttk/koller/model_data/model_data_TTBar_10M_train_ev_30_ep/results_qcd_samples.npz')['probs']

data_wqq_JetClass=np.load('/net/data_ttk/koller/model_data/model_data_WToQQ_10M_train_ev_30_ep/results_jetclass.npz')['probs']
data_wqq_w_samples=np.load('/net/data_ttk/koller/model_data/model_data_WToQQ_10M_train_ev_30_ep/results_w_samples.npz')['probs']
data_wqq_qcd_samples=np.load('/net/data_ttk/koller/model_data/model_data_WToQQ_10M_train_ev_30_ep/results_qcd_samples.npz')['probs']

data_zqq_JetClass=np.load('/net/data_ttk/koller/model_data/model_data_ZToQQ_10M_train_ev_30_ep/results_jetclass.npz')['probs']
data_zqq_z_samples=np.load('/net/data_ttk/koller/model_data/model_data_ZToQQ_10M_train_ev_30_ep/results_z_data.npz')['probs']
data_zqq_qcd_samples=np.load('/net/data_ttk/koller/model_data/model_data_ZToQQ_10M_train_ev_30_ep/results_qcd_samples.npz')['probs']

data_hgg_JetClass=np.load('/net/data_ttk/koller/model_data/model_data_HToGG_10M_train_ev_30_ep/results_jetclass.npz')['probs']
data_hgg_h_samples=np.load('/net/data_ttk/koller/model_data/model_data_HToGG_10M_train_ev_30_ep/results_h_samples.npz')['probs']
data_hgg_qcd_samples=np.load('/net/data_ttk/koller/model_data/model_data_HToGG_10M_train_ev_30_ep/results_qcd_samples.npz')['probs']

s_top_qcd_qcd=data_ttbar_qcd_samples-data_qcd_qcd_samples
s_zqq_qcd_qcd=data_zqq_qcd_samples-data_qcd_qcd_samples
s_wqq_qcd_qcd=data_wqq_qcd_samples-data_qcd_qcd_samples
s_hgg_qcd_qcd=data_hgg_qcd_samples-data_qcd_qcd_samples
s_top_qcd_top=data_ttbar_top_samples-data_qcd_top_samples
s_zqq_qcd_z=data_zqq_z_samples-data_qcd_z_samples
s_wqq_qcd_w=data_wqq_w_samples-data_qcd_w_samples
s_hgg_qcd_h=data_hgg_h_samples-data_qcd_h_samples

#plot_ll_dist([data_hgg_JetClass,data_hgg_h_samples,data_ttbar_top_samples,data_ttbar_JetClass,data_wqq_w_samples,data_wqq_JetClass,data_zqq_z_samples,data_zqq_JetClass,data_qcd_qcd_samples,data_qcd_JetClass],'JetClass log-likelihood distributions','_jetclass_dist',0)
#plot_ll_dist([data_ttbar_top_samples,data_ttbar_qcd_samples,data_qcd_qcd_samples,data_qcd_top_samples],'samples log-likelihood distributions','_top_qcd_samp_dist',1)
#plot_ll_dist([data_zqq_z_samples,data_zqq_qcd_samples,data_qcd_qcd_samples,data_qcd_z_samples],'samples log-likelihood distributions','_z_qcd_samp_dist',2)
#plot_ll_dist([data_wqq_w_samples,data_wqq_qcd_samples,data_qcd_qcd_samples,data_qcd_w_samples],'samples log-likelihood distributions','_w_qcd_samp_dist',3)
#plot_ll_dist([data_hgg_h_samples,data_hgg_qcd_samples,data_qcd_qcd_samples,data_qcd_h_samples],'samples log-likelihood distributions','_h_qcd_samp_dist',4)

plot_llr_roc([s_top_qcd_top,s_top_qcd_qcd],['top','qcd'],'top_qcd',0)
plot_llr_roc([s_zqq_qcd_z,s_zqq_qcd_qcd],['zqq','qcd'],'zqq_qcd',1)
plot_llr_roc([s_wqq_qcd_w,s_wqq_qcd_qcd],['wqq','qcd'],'wqq_qcd',2)
plot_llr_roc([s_hgg_qcd_h,s_hgg_qcd_qcd],['hgg','qcd'],'hgg_qcd',3)
