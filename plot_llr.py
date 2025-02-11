import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
from scipy import stats
from tqdm import trange
from sklearn.metrics import roc_curve,roc_auc_score


def plot_ll_dist(results,title,name,label_num):
    label=[['top JetClass','qcd JetClass','top samples','qcd samples','qcd samples, ttbar model','top samples, qcd model'],['top JetClass','zqq JetClass','qcd JetClass','wqq JetClass'],['zqq JetClass','wqq JetClass','zqq samples','wqq samples','wqq samples, zqq model','zqq samples, wqq model']]
    label=label[label_num]
    color=['darkslateblue','orangered','steelblue','indianred','mediumvioletred','midnightblue']
    alpha=[0.5,0.5,0.5,0.5,0.8,0.8,0.5,0.8]
    plt.figure(figsize=(10,4))
    plt.grid()
    htype=['bar','bar','bar','bar','step','step','bar','step']
    n_bins=150
    for i in range(len(results)):
        #res_fit=stats.norm.fit(results[i])
        #print(res_fit)
        plt.hist(results[i],bins=n_bins,histtype=htype[i],density=True,label=label[i],color=color[i],alpha=alpha[i])
        #n,bins,patches=plt.hist(results[i],bins=n_bins,histtype=htype[i],density=True,label=label[i],color=color[i],alpha=alpha[i])
        #y=stats.norm.pdf(bins,res_fit[0],res_fit[1])
        #plt.close()
        #l=plt.plot(bins,y)
    plt.xlim(-800,0)
    plt.xlabel('log-likelihood')
    plt.ylabel('')
    plt.legend()
    plt.title(title)
    plt.savefig('plots_llr/plot_ll_1Mevents_'+name+'.png')

def plot_llr(data,label,name):
    plt.figure(figsize=(10,4))
    plt.hist(data[0]-data[1],bins=400,color='steelblue',label=label[0],alpha=0.6,density=True)
    plt.hist(data[2]-data[3],bins=400,color='indianred',label=label[1],alpha=0.6,density=True)
    plt.legend()
    plt.xlim((-45,30))
    plt.xlabel('log-likelihood ratio')
    plt.ylabel('normalized distribution')
    plt.savefig('plots_llr/'+name)

def plot_roc_auc(llr,title,name):
    labels=np.append(np.zeros(len(llr[0])),np.ones(len(llr[1])))
    predictions=np.append(llr[0],llr[1])
    roc=roc_curve(labels,predictions)
    plt.figure()
    plt.scatter(roc[1],1/roc[0],color='orangered',marker='.',zorder=3,s=5)
    plt.xlim((0,1))
    plt.yscale('log')
    plt.xlabel('$\epsilon_{top}$')
    plt.ylabel('$1/\epsilon_{QCD}$')
    plt.title('ROC-curve '+title)

    auc=roc_auc_score(labels,predictions)
    print('auc: ',auc)
    plt.text(0.6,13000,'auc: '+str(auc))
    plt.grid(which='both',color='grey',lw=0.5)
    plt.savefig('plots_llr/'+name)

data_ttbar_JetClass=np.load('/net/data_ttk/koller/model_data_TTBar_10M_train_ev_30_ep/results_test.npz')['probs']
data_ttbar_top_samples=np.load('/net/data_ttk/koller/model_data_TTBar_10M_train_ev_30_ep/results_top_samples.npz')['probs']
data_ttbar_qcd_samples=np.load('/net/data_ttk/koller/model_data_TTBar_10M_train_ev_30_ep/results_qcd_samples.npz')['probs']
data_ttbar_w_samples=np.load('/net/data_ttk/koller/model_data_TTBar_10M_train_ev_30_ep/results_w_samples.npz')['probs']
#data_ttbar_z_samples=np.load('/net/data_ttk/koller/model_data_TTBar_10M_train_ev_30_ep/results_z_samples.npz')['probs']

data_znn_JetClass=np.load('/net/data_ttk/koller/model_data_ZJetsToNuNu_10M_train_ev_30_ep/results_test.npz')['probs']
data_znn_qcd_samples=np.load('/net/data_ttk/koller/model_data_ZJetsToNuNu_10M_train_ev_30_ep/results_qcd_samples.npz')['probs']
data_znn_top_samples=np.load('/net/data_ttk/koller/model_data_ZJetsToNuNu_10M_train_ev_30_ep/results_top_samples.npz')['probs']
#data_znn_w_samples=np.load('/net/data_ttk/koller/model_data_ZJetsToNuNu_10M_train_ev_30_ep/results_w_samples.npz')['probs']
#data_znn_z_samples=np.load('/net/data_ttk/koller/model_data_ZJetsToNuNu_10M_train_ev_30_ep/results_z_samples.npz')['probs']

data_wqq_JetClass=np.load('/net/data_ttk/koller/model_data_WToQQ_10M_train_ev_30_ep/results_test.npz')['probs']
data_wqq_w_samples=np.load('/net/data_ttk/koller/model_data_WToQQ_10M_train_ev_30_ep/results_w_samples.npz')['probs']
data_wqq_z_samples=np.load('/net/data_ttk/koller/model_data_WToQQ_10M_train_ev_30_ep/results_z_samples.npz')['probs']
data_wqq_top_samples=np.load('/net/data_ttk/koller/model_data_WToQQ_10M_train_ev_30_ep/results_top_samples.npz')['probs']
data_wqq_qcd_samples=np.load('/net/data_ttk/koller/model_data_WToQQ_10M_train_ev_30_ep/results_qcd_samples.npz')['probs']

data_zqq_JetClass=np.load('/net/data_ttk/koller/model_data_ZToQQ_10M_train_ev_30_ep/results_jetclass.npz')['probs']
data_zqq_z_samples=np.load('/net/data_ttk/koller/model_data_ZToQQ_10M_train_ev_30_ep/results_z_data.npz')['probs']
data_zqq_w_samples=np.load('/net/data_ttk/koller/model_data_ZToQQ_10M_train_ev_30_ep/results_w_data.npz')['probs']
#data_zqq_qcd_samples=np.load('/net/data_ttk/koller/model_data_ZToQQ_10M_train_ev_30_ep/results_qcd_samples.npz')['probs']
#data_zqq_top_samples=np.load('/net/data_ttk/koller/model_data_ZToQQ_10M_train_ev_30_ep/results_top_samples.npz')['probs']


plot_ll_dist([data_ttbar_JetClass,data_znn_JetClass,data_ttbar_top_samples,data_znn_qcd_samples,data_ttbar_qcd_samples,data_znn_top_samples],'samples log-likelihood distributions','_top_qcd_samp_dist',0)
plot_ll_dist([data_ttbar_JetClass,data_zqq_JetClass,data_znn_JetClass,data_wqq_JetClass],'JetClass log-likelihood distributions','_jetclass_dist',1)
plot_ll_dist([data_zqq_JetClass,data_wqq_JetClass,data_zqq_z_samples,data_wqq_w_samples,data_zqq_w_samples,data_wqq_z_samples],'samples log-likelihood distributions','_z_w_samp_dist',2)
plot_llr([data_ttbar_top_samples,data_znn_top_samples,data_ttbar_qcd_samples,data_znn_qcd_samples],['top','qcd'],'plot_llr_top_qcd')
plot_llr([data_zqq_z_samples,data_wqq_z_samples,data_zqq_w_samples,data_wqq_w_samples],['zqq','wqq'],'plot_llr_zqq_wqq')
plot_roc_auc([(data_ttbar_qcd_samples-data_znn_qcd_samples),(data_ttbar_top_samples-data_znn_top_samples)],'top/QCD','plot_roc_top_qcd.png')
plot_roc_auc([(data_zqq_w_samples-data_wqq_w_samples),(data_zqq_z_samples-data_wqq_z_samples)],'zqq/wqq','plot_roc_zqq_wqq.png')
