import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def GetDataFromQCDT(file_dir,file_name_qcd,file_name_top):

    
    

    file_qcd=file_dir+'/'+file_name_qcd
    evalprob_qcdfromqcd = np.load(file_qcd)
    
    file_top=file_dir+'/'+file_name_top
    evalprob_topfromqcd = np.load(file_top)



    return evalprob_qcdfromqcd,evalprob_topfromqcd


def GetDataFromTopT(file_dir,file_name_top,file_name_qcd):

    

    file_top=file_dir+'/'+file_name_top
    evalprob_topfromtop = np.load(file_top)
    
    file_qcd=file_dir+'/'+file_name_qcd
    evalprob_qcdfromtop = np.load(file_qcd)
    
    return evalprob_topfromtop,evalprob_qcdfromtop


def ExpectedProb(evalprob):
    exp_logp=np.log(1/(39402))*evalprob['n_const']
    return exp_logp

def ComputeLLR(evalprobT,evalprobF,type):

    s=evalprobT['probs']-evalprobF['probs']

    return s

def PlotLLR(s_qcd,s_top,path_to_plots,plot_title):
    bins = np.linspace(-340, 340, 40)
    plt.hist(s_qcd,histtype='step',bins=bins,density=True,color='blue',label='QCD')
    plt.hist(s_top,histtype='step',bins=bins,density=True,color='black',label='Top')
    plt.legend()
    plt.title(plot_title,loc='left')
    plt.savefig(path_to_plots+'/plot_LLR_test_1.png')
    plt.close()
    return


def ROCcurve(s_qcd,s_top,path_to_plots,plot_title):
    from sklearn.metrics import roc_curve, roc_auc_score
    fpr, tpr, _ = roc_curve(np.append(np.zeros(len(s_qcd)), np.ones(len(s_top))), np.append(s_qcd, s_top))
    #fpr, tpr, _ = roc_curve(np.append(np.zeros(len(s_top)), np.ones(len(s_qcd))), np.append(s_top, s_qcd))
    #fpr, tpr, _ = roc_curve(np.append(np.ones(len(s_qcd)), np.zeros(len(s_top))), np.append(s_qcd, s_top))
    print(roc_auc_score(np.append(np.zeros(len(s_qcd)), np.ones(len(s_top))), np.append(s_qcd, s_top)))
    auc_score=roc_auc_score(np.append(np.zeros(len(s_qcd)), np.ones(len(s_top))), np.append(s_qcd, s_top))
    #auc_score=roc_auc_score(np.append(np.ones(len(s_qcd)), np.zeros(len(s_top))), np.append(s_qcd, s_top))
    print(fpr)
    plt.plot(fpr,tpr, label="Transformer", c="r")
    plt.legend()
    plt.title(plot_title+' -- auc='+str(auc_score),loc='left')
    plt.savefig(path_to_plots+'/plot_ROC_1.png')
    plt.close()
    
    plt.plot(tpr,1/fpr, label="Transformer", c="r")
    plt.ylim(1, 1e6)
    plt.xlim(0, 1)
    plt.xlabel(r"$\epsilon_{\rm{top}}$")
    plt.ylabel(r"$1 / \epsilon_{\rm{QCD}}$")
    plt.yscale('log')
    #plt.xscale('log')
    plt.legend()
    plt.title(plot_title+' -- auc='+str(auc_score),loc='left')
    plt.savefig(path_to_plots+'/plot_ROC2_1.png')
    plt.close()
    return



def plot_probs(evalprob,path_to_plots,plot_title):

 plt.hist(evalprob['probs'],histtype='step',bins=30,density=True,color='blue')

 exp_logp=ExpectedProb(evalprob)

 plt.hist(exp_logp,histtype='step',bins=30,linestyle='--',density=True,color='blue')
 plt.title(plot_title,loc='left')
 plt.savefig(path_to_plots+'/plot_probs_test_1.png')
 plt.close()

 return



def plot_probs(evalprob_qcdfromqcd,evalprob_topfromqcd,evalprob_topfromtop,evalprob_qcdfromtop,path_to_plots,plot_title):

 plt.hist(evalprob_qcdfromqcd['probs'],histtype='step',bins=30,density=True,color='blue',label='QCD')
 plt.hist(evalprob_qcdfromtop['probs'],histtype='step',bins=30,density=True,color='blue',linestyle='--')
 
 plt.hist(evalprob_topfromtop['probs'],histtype='step',bins=30,density=True,color='black',label='Top')
 plt.hist(evalprob_topfromqcd['probs'],histtype='step',bins=30,density=True,color='black',linestyle='--')
 plt.xlabel('log(p)')
 plt.legend()
 plt.title(plot_title,loc='left')
 plt.savefig(path_to_plots+'/plot_probs_test_1.png')
 plt.close()

 return



def plot_color(evalprob,path_to_plots,plot_title):

 plt.plot(evalprob['probs'],evalprob['n_const'],'.')
 plt.xlabel('n_{const}')
 plt.ylabel('log(p)')
 plt.title(plot_title,loc='left')
 plt.savefig(path_to_plots+'/plot_plot_test_1.png')
 plt.close()

 return
 
 
def plot_contour(evalprob_qcdfromqcd,evalprob_topfromtop,path_to_plots,plot_title):

 sns.kdeplot(data=evalprob_qcdfromqcd,x='probs',y='n_const',levels=4,color='blue', fill=True,alpha=.5)
 sns.kdeplot(data=evalprob_topfromtop,x='probs',y='n_const',levels=4,color='black', fill=True,alpha=.5)
 plt.plot(np.log(1. / 39402) * np.linspace(0, 100, 100), np.linspace(0, 100, 100), linestyle="--", color='grey')
 plt.ylabel('$n_{const}$')
 plt.xlabel('log(p)')
 plt.title(plot_title,loc='left')
 plt.savefig(path_to_plots+'/plot_contour_test_1.png')
 plt.close()

 return

test_results_dir='../LLRModels/'


qcd_file_dir=test_results_dir+'/ZJetsToNuNu_run_scan_10M_N1G96CW/'
qcd_file_name_qcd='results_test_eval_optclass_testset_nsamples200000.npz'
qcd_file_name_top='results_test_eval_optclass_testset_other_nsamples200000.npz'


top_file_dir=test_results_dir+'/TTBar_run_testwall_10M_11/'
top_file_name_top='results_test_eval_optclass_testset_nsamples200000.npz'
top_file_name_qcd='results_test_eval_optclass_testset_other_nsamples200000.npz'
    

'''
qcd_file_dir=test_results_dir+'/zjettonunu_run_b_7/'
qcd_file_name_qcd='results_test_eval_200k.npz'
qcd_file_name_top='results_test_eval_other_200k.npz'


top_file_dir=test_results_dir+'/TTBar/ttbar_run_b_2_6/'
top_file_name_top='results_test_eval_200k.npz'
top_file_name_qcd='results_test_eval_other_200k.npz'
'''
plot_title='LLR generated events'
#plot_title='JetClass test events'

path_to_plots=test_results_dir+'/test_model1_ttbar_qcd_200k/'
os.makedirs(path_to_plots,exist_ok=True)


evalprob_qcdfromqcd,evalprob_topfromqcd=GetDataFromQCDT(qcd_file_dir,qcd_file_name_qcd,qcd_file_name_top)
evalprob_topfromtop,evalprob_qcdfromtop=GetDataFromTopT(top_file_dir,top_file_name_top,top_file_name_qcd)

plot_probs(evalprob_qcdfromqcd,evalprob_topfromqcd,evalprob_topfromtop,evalprob_qcdfromtop,path_to_plots,plot_title)
plot_contour(evalprob_qcdfromqcd,evalprob_topfromtop,path_to_plots,plot_title)


s_top=ComputeLLR(evalprob_topfromtop,evalprob_topfromqcd,'qcd')
s_qcd=ComputeLLR(evalprob_qcdfromtop,evalprob_qcdfromqcd,'qcd')
PlotLLR(s_qcd,s_top,path_to_plots,plot_title)
ROCcurve(s_qcd,s_top,path_to_plots,plot_title)

exit()
plot_color(evalprob)
plot_contour(evalprob)
