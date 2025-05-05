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
    plt.savefig(f'plots_llr/plot_ll_1Mevents_{name}.svg')

def find_R_val(roc_data,val):
    R_val=np.interp(val,roc_data[1],1/roc_data[0])
    return R_val

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
    plt.savefig(f'plots_llr/plot_llr_{name}.svg')

    labels=np.append(np.zeros(len(data[1])),np.ones(len(data[0])))
    predictions=np.append(data[1],data[0])
    roc=roc_curve(labels,predictions)
    auc=roc_auc_score(labels,predictions)
    R_30_ind=np.argmin(np.abs(roc[1]-0.3))
    print(roc[1][R_30_ind])
    R_30=1/roc[0][R_30_ind]
    R_50_ind=np.argmin(np.abs(roc[1]-0.5))
    print(roc[1][R_50_ind])
    R_50=1/roc[0][R_50_ind]
    R_30,R_50=find_R_val(roc,[0.3,0.5])
    #print('\nrejection values:',R_30,R_50,R_30_intp,R_50_intp)
    plt.figure()
    plt.scatter(roc[1],1/roc[0],color='orangered',marker='.',zorder=3,s=5)
    plt.scatter(roc[1][R_30_ind],R_30,color='black',marker='.',label='$R_{30}=$'+str(round(R_30,5)),zorder=5)
    plt.scatter(roc[1][R_50_ind],R_50,color='black',marker='.',label='$R_{50}=$'+str(round(R_50,5)),zorder=5)
    plt.hlines([R_30,R_50],[0,0],[0.3,0.5],color='black',linestyle='dashed',linewidth=1,zorder=6)
    plt.vlines([0.3,0.5],[0,0],[R_30,R_50],color='black',linestyle='dashed',linewidth=1,zorder=6)
    plt.scatter(0,0,color='white',label='auc: '+str(auc))
    plt.xlim((0,1))
    plt.yscale('log')
    plt.xlabel('$\epsilon_{top}$')
    plt.ylabel('$1/\epsilon_{QCD}$')
    plt.legend()
    plt.title(f'ROC-curve {label[0]}/{label[1]}')

    #print('auc: ',auc)
    #plt.text(0.6,13000,'auc: '+str(auc))
    plt.grid(which='both',color='grey',lw=0.5)
    plt.savefig(f'plots_llr/plot_roc_{name}.svg')

def plot_roc_comp(data_opt,data_clsf,data_clsf_best,label,name):
    labels=np.append(np.zeros(len(data_opt[1])),np.ones(len(data_opt[0])))
    predictions=np.append(data_opt[1],data_opt[0])
    roc_opt=roc_curve(labels,predictions)
    auc_opt=roc_auc_score(labels,predictions)
    roc_clsf=roc_curve(data_clsf['labels'],data_clsf['predictions'])
    auc_clsf=roc_auc_score(data_clsf['labels'],data_clsf['predictions'])
    #roc_clsf_best=roc_curve(data_clsf_best['labels'],data_clsf_best['preds'])
    #auc_clsf_best=roc_auc_score(data_clsf_best['labels'],data_clsf_best['preds'])
    R_50_opt_ind=np.argmin(np.abs(roc_opt[1]-0.5))
    R_50_clsf_ind=np.argmin(np.abs(roc_clsf[1]-0.5))
    #R_50_clsf_best_ind=np.argmin(np.abs(roc_clsf_best[1]-0.5))
    #print(R_50_ind,roc[1][R_50_ind-1],roc[1][R_50_ind],roc[1][R_50_ind+1])
    R_50_opt=1/roc_opt[0][R_50_opt_ind]
    R_50_clsf=1/roc_clsf[0][R_50_clsf_ind]
    R_30_opt,R_50_opt=find_R_val(roc_opt,[0.3,0.5])
    R_30_clsf,R_50_clsf=find_R_val(roc_clsf,[0.3,0.5])
    #R_30_clsf_best,R_50_clsf_best=find_R_val(roc_clsf_best,[0.3,0.5])

    plt.figure()
    plt.scatter(roc_opt[1],1/roc_opt[0],color='orangered',marker='.',zorder=3,s=5,label=f'optimal, $R_{{30}}$: {round(R_30_opt,4)}, $R_{{50}}$: {round(R_50_opt,4)}\n    auc: {round(auc_opt,4)}')
    #plt.scatter(0,0,color='white',alpha=0,label='auc optimal: '+str(round(auc_opt,5)))
    #plt.scatter(roc_clsf_best[1],1/roc_clsf_best[0],color='magenta',marker='.',zorder=3,s=5,label=f'classifier best, $R_{{30}}$: {round(R_30_clsf_best,4)}, $R_{{50}}$: {round(R_50_clsf_best,4)}')
    #plt.scatter(0,0,color='white',alpha=0,label='auc classifier best: '+str(round(auc_clsf_best,5)))
    plt.scatter(roc_clsf[1],1/roc_clsf[0],color='royalblue',marker='.',zorder=3,s=5,label=f'classifier, $R_{{30}}$: {round(R_30_clsf,4)}, $R_{{50}}$: {round(R_50_clsf,4)}\n    auc: {round(auc_clsf,4)}')
    #plt.scatter(0,0,color='white',alpha=0,label='auc classifier: '+str(round(auc_clsf,5)))
    plt.scatter([0.3,0.3],[R_30_clsf,R_30_opt],color='black',marker='.',zorder=5) #,label='$R_{30}=$'+str(round(R_30,5))
    plt.scatter([0.5,0.5],[R_50_clsf,R_50_opt],color='black',marker='.',zorder=5) #label='$R_{30}=$'+str(round(R_30,5)),
    plt.hlines([R_30_opt,R_50_opt,R_30_clsf,R_50_clsf],[0,0,0,0],[0.3,0.5,0.3,0.5],color='black',linestyle='dashed',linewidth=1,zorder=6)
    plt.vlines([0.3,0.5],[0,0],[max(R_30_opt,R_30_clsf),max(R_50_opt,R_50_clsf)],color='black',linestyle='dashed',linewidth=1,zorder=6)
    plt.xlim((0,1))
    plt.yscale('log')
    plt.xlabel('$\epsilon_{top}$')
    plt.ylabel('$1/\epsilon_{QCD}$')
    plt.legend(loc=1,prop={'size': 8})
    plt.title(f'ROC-curve {label[0]}/{label[1]}')
    plt.grid(which='both',color='grey',lw=0.5)
    plt.savefig(f'plots_llr/plot_roc_{name}_opt_class.svg')
    plt.savefig(f'plots_llr/plot_roc_{name}_opt_class.png')

    return

def roc_curve_comp_const(data,data_qcd,label,name):
    l=len(data)
    auc_list=np.empty(l)
    colors=plt.cm.rainbow_r(np.linspace(0,1,l))
    plt.figure(figsize=(10,8))
    for i in range(l):
        labels=np.append(np.zeros(len(data_qcd[i])),np.ones(len(data[i])))
        predictions=np.append(data_qcd[i],data[i])
        roc=roc_curve(labels,predictions)
        auc=roc_auc_score(labels,predictions)
        auc_list[i]=auc
        R_30,R_50=find_R_val(roc,[0.3,0.5])
        plt.scatter(roc[1],1/roc[0],color=colors[i],marker='.',zorder=3,s=5,label=f'\# constituents: {label[i]}\n$R_{{50}}$: {round(R_50,4)}')  # $R_{{30}}$: {round(R_30,4)}, 
        plt.scatter([0.3,0.5],[R_30,R_50],color='black',marker='.',zorder=5,s=10)
        #plt.scatter(0.5,R_50,color='black',marker='.',label='$R_{50}=$'+str(round(R_50,5)),zorder=5)
        plt.hlines([R_30,R_50],[0,0],[0.3,0.5],color='black',linestyle='dashed',linewidth=0.5,zorder=6)
        plt.vlines([0.3,0.5],[0,0],[R_30,R_50],color='black',linestyle='dashed',linewidth=0.5,zorder=6)
        plt.scatter(0,0,color='white',alpha=0,label=f'auc: {round(auc,4)}')
    plt.xlim((0,1))
    plt.yscale('log')
    plt.xlabel('$\epsilon_{top}$')
    plt.ylabel('$1/\epsilon_{QCD}$')
    plt.legend(loc='upper right')
    plt.title(f'ROC-curve {name}/QCD')
    plt.grid(which='both',color='grey',lw=0.5)
    plt.savefig(f'plots_llr/plot_roc_comp_const_{name}_QCD_200k_samp.svg')
    plt.savefig(f'plots_llr/plot_roc_comp_const_{name}_QCD_200k_samp.png')
    return

def plot_roc_comp_classes(data_opt,data_clsf,label):
    l=len(data_opt)
    auc_list=np.empty(l)
    colors=plt.cm.Set1(np.linspace(0,1,l+1))
    plt.figure(figsize=(10,7))
    for i in range(l):
        labels=np.append(np.zeros(len(data_opt[i][1])),np.ones(len(data_opt[i][0])))
        predictions=np.append(data_opt[i][1],data_opt[i][0])
        roc_opt=roc_curve(labels,predictions)
        auc_opt=roc_auc_score(labels,predictions)
        roc_clsf=roc_curve(data_clsf[i]['labels'],data_clsf[i]['predictions'])
        auc_clsf=roc_auc_score(data_clsf[i]['labels'],data_clsf[i]['predictions'])
        #auc_list[i]=auc
        R_30_opt,R_50_opt=find_R_val(roc_opt,[0.3,0.5])
        R_30_clsf,R_50_clsf=find_R_val(roc_clsf,[0.3,0.5])

        plt.plot(roc_opt[1],1/roc_opt[0],color=colors[i],zorder=3,label=f'{label[i]} optimal: $R_{{30}}$: {round(R_30_opt,4)}, $R_{{50}}$: {round(R_50_opt,4)}\n classifier: $R_{{30}}$: {round(R_30_clsf,4)}, $R_{{50}}$: {round(R_50_clsf,4)}')       #,marker='.',s=5
        plt.scatter([0.3,0.5],[R_30_opt,R_50_opt],color='black',marker='.',zorder=5,s=20)
        plt.scatter(0,0,color='white',alpha=0,label=f'auc optimal: {auc_opt}\nauc classifier: {auc_clsf}')
        plt.plot(roc_clsf[1],1/roc_clsf[0],color=colors[i],linestyle='dashed',zorder=3)       #,marker='.',s=5,label=f'{label[i]},'
        plt.scatter([0.3,0.5],[R_30_opt,R_50_opt],color='black',marker='.',zorder=5,s=20)
        #plt.scatter(0,0,color='white',alpha=0,label=f'auc: {auc_opt}')

        #plt.hlines([R_30_opt,R_50_opt,R_30_clsf,R_50_clsf],[0,0,0,0],[0.3,0.5,0.3,0.5],color='black',linestyle='dashed',linewidth=0.7,zorder=6)
        #plt.vlines([0.3,0.5,0.3,0.5],[0,0,0,0],[R_30_opt,R_50_opt,R_30_clsf,R_50_clsf],color='black',linestyle='dashed',linewidth=0.7,zorder=6)
    plt.xlim((0,1))
    plt.yscale('log')
    plt.xlabel('$\epsilon$')    #_{top}
    plt.ylabel('$1/\epsilon_{QCD}$')
    plt.legend()
    plt.title(f'ROC-curve') # {name}/QCD
    plt.grid(which='both',color='grey',lw=0.5)
    plt.savefig(f'plots_llr/plot_roc_comp_classes_QCD.svg')
    plt.savefig(f'plots_llr/plot_roc_comp_classes_QCD.png')
    return

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
data_zqq_z_samples=np.load('/net/data_ttk/koller/model_data/model_data_ZToQQ_10M_train_ev_30_ep/results_z_samples.npz')['probs']
data_zqq_qcd_samples=np.load('/net/data_ttk/koller/model_data/model_data_ZToQQ_10M_train_ev_30_ep/results_qcd_samples.npz')['probs']

data_hgg_JetClass=np.load('/net/data_ttk/koller/model_data/model_data_HToGG_10M_train_ev_30_ep/results_jetclass.npz')['probs']
data_hgg_h_samples=np.load('/net/data_ttk/koller/model_data/model_data_HToGG_10M_train_ev_30_ep/results_h_samples.npz')['probs']
data_hgg_qcd_samples=np.load('/net/data_ttk/koller/model_data/model_data_HToGG_10M_train_ev_30_ep/results_qcd_samples.npz')['probs']

#predictions classifier
pred_class_hgg=np.load('/net/data_ttk/koller/class_data/class_hgg_qcd_10M_ev_30_ep/predictions_test.npz')
pred_class_hgg_best=np.load('/net/data_ttk/koller/class_data/class_hgg_qcd_10M_ev_30_ep/preds_best.npz')
pred_class_wqq=np.load('/net/data_ttk/koller/class_data/class_wqq_qcd_10M_ev_30_ep/predictions_test.npz')
pred_class_wqq_best=np.load('/net/data_ttk/koller/class_data/class_wqq_qcd_10M_ev_30_ep/preds_best.npz')
pred_class_zqq=np.load('/net/data_ttk/koller/class_data/class_zqq_qcd_10M_ev_30_ep/predictions_test.npz')
pred_class_zqq_best=np.load('/net/data_ttk/koller/class_data/class_zqq_qcd_10M_ev_30_ep/preds_best.npz')
pred_class_top=np.load('/net/data_ttk/koller/class_data/class_top_qcd_1M_ev_30_ep/predictions_test.npz')
pred_class_top_best=np.load('/net/data_ttk/koller/class_data/class_top_qcd_1M_ev_30_ep/preds_best.npz')

s_top_qcd_qcd=data_ttbar_qcd_samples-data_qcd_qcd_samples
s_zqq_qcd_qcd=data_zqq_qcd_samples-data_qcd_qcd_samples
s_wqq_qcd_qcd=data_wqq_qcd_samples-data_qcd_qcd_samples
s_hgg_qcd_qcd=data_hgg_qcd_samples-data_qcd_qcd_samples
s_top_qcd_top=data_ttbar_top_samples-data_qcd_top_samples
s_zqq_qcd_z=data_zqq_z_samples-data_qcd_z_samples
s_wqq_qcd_w=data_wqq_w_samples-data_qcd_w_samples
s_hgg_qcd_h=data_hgg_h_samples-data_qcd_h_samples

'''
plot_ll_dist([data_hgg_JetClass,data_hgg_h_samples,data_ttbar_top_samples,data_ttbar_JetClass,data_wqq_w_samples,data_wqq_JetClass,data_zqq_z_samples,data_zqq_JetClass,data_qcd_qcd_samples,data_qcd_JetClass],'JetClass log-likelihood distributions','_jetclass_dist',0)
plot_ll_dist([data_ttbar_top_samples,data_ttbar_qcd_samples,data_qcd_qcd_samples,data_qcd_top_samples],'samples log-likelihood distributions','_top_qcd_samp_dist',1)
plot_ll_dist([data_zqq_z_samples,data_zqq_qcd_samples,data_qcd_qcd_samples,data_qcd_z_samples],'samples log-likelihood distributions','_z_qcd_samp_dist',2)
plot_ll_dist([data_wqq_w_samples,data_wqq_qcd_samples,data_qcd_qcd_samples,data_qcd_w_samples],'samples log-likelihood distributions','_w_qcd_samp_dist',3)
plot_ll_dist([data_hgg_h_samples,data_hgg_qcd_samples,data_qcd_qcd_samples,data_qcd_h_samples],'samples log-likelihood distributions','_h_qcd_samp_dist',4)

plot_llr_roc([s_top_qcd_top,s_top_qcd_qcd],['top','qcd'],'top_qcd',0)
plot_llr_roc([s_zqq_qcd_z,s_zqq_qcd_qcd],['zqq','qcd'],'zqq_qcd',1)
plot_llr_roc([s_wqq_qcd_w,s_wqq_qcd_qcd],['wqq','qcd'],'wqq_qcd',2)
plot_llr_roc([s_hgg_qcd_h,s_hgg_qcd_qcd],['hgg','qcd'],'hgg_qcd',3)'''

plot_roc_comp([s_hgg_qcd_h,s_hgg_qcd_qcd],pred_class_hgg,pred_class_hgg_best,['hgg','qcd'],'hgg_qcd_10M')
plot_roc_comp([s_wqq_qcd_w,s_wqq_qcd_qcd],pred_class_wqq,pred_class_wqq_best,['wqq','qcd'],'wqq_qcd_10M')
plot_roc_comp([s_zqq_qcd_z,s_zqq_qcd_qcd],pred_class_zqq,pred_class_zqq_best,['zqq','qcd'],'zqq_qcd_10M')
plot_roc_comp([s_top_qcd_top,s_top_qcd_qcd],pred_class_top,pred_class_top_best,['top','qcd'],'top_qcd_10M')

#plot_roc_comp_classes([[s_top_qcd_top,s_top_qcd_qcd],[s_hgg_qcd_h,s_hgg_qcd_qcd],[s_wqq_qcd_w,s_wqq_qcd_qcd],[s_zqq_qcd_z,s_zqq_qcd_qcd]],[pred_class_top,pred_class_hgg,pred_class_wqq,pred_class_zqq],['top','hgg','wqq','zqq'])