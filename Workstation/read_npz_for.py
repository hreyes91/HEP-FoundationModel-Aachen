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
    bins = np.linspace(-100, 120, 40)
    plt.hist(s_qcd,histtype='step',bins=bins,density=True,color='blue',label='QCD')
    plt.hist(s_top,histtype='step',bins=bins,density=True,color='black',label='Top')
    plt.legend()
    plt.title(plot_title,loc='left')
    plt.savefig(path_to_plots+'/plot_LLR_test_1.png')
    plt.close()
    return


def ROCcurve(s_qcd_5m,s_top_5m,s_qcd_2m,s_top_2m,path_to_plots,plot_title):
    from sklearn.metrics import roc_curve, roc_auc_score
    fpr_2m, tpr_2m, _2m = roc_curve(np.append(np.zeros(len(s_qcd_2m)), np.ones(len(s_top_2m))), np.append(s_qcd_2m, s_top_2m))
    print(roc_auc_score(np.append(np.zeros(len(s_qcd_2m)), np.ones(len(s_top_2m))), np.append(s_qcd_2m, s_top_2m)))
    print(fpr_2m)
    plt.plot(fpr_2m,tpr_2m, label="Transformer 2M", c="r")
    
    fpr_5m, tpr_5m, _5m = roc_curve(np.append(np.zeros(len(s_qcd_5m)), np.ones(len(s_top_5m))), np.append(s_qcd_5m, s_top_5m))
    print(roc_auc_score(np.append(np.zeros(len(s_qcd_5m)), np.ones(len(s_top_5m))), np.append(s_qcd_5m, s_top_5m)))
    print(fpr_5m)
    plt.plot(fpr_5m,tpr_5m, label="Transformer 5M", c="b")

    
    plt.legend()
    plt.title(plot_title,loc='left')
    plt.savefig(path_to_plots+'/plot_ROC.png')
    plt.close()
    
    
    plt.plot(tpr_2m,1/fpr_2m, label="Transformer 2M", c="r")
    plt.plot(tpr_5m,1/fpr_5m, label="Transformer 5M", c="b")
    plt.ylim(1, 1e4)
    plt.xlim(0, 1)
    plt.xlabel(r"$\epsilon_{\rm{top}}$")
    plt.ylabel(r"$1 / \epsilon_{\rm{QCD}}$")
    plt.yscale('log')
    #plt.xscale('log')
    plt.legend()
    plt.title(plot_title,loc='left')
    plt.savefig(path_to_plots+'/plot_ROC2.png')
    plt.close()
    return

'''

def plot_probs(evalprob,path_to_plots,plot_title):

 plt.hist(evalprob['probs'],histtype='step',bins=30,density=True,color='blue')

 exp_logp=ExpectedProb(evalprob)

 plt.hist(exp_logp,histtype='step',bins=30,linestyle='--',density=True,color='blue')
 plt.title(plot_title,loc='left')
 plt.savefig(path_to_plots+'/plot_probs_test_1.png')
 plt.close()

 return


'''
def plot_probs(evalprob_qcdfromqcd_5m,evalprob_topfromqcd_5m,evalprob_topfromtop_5m,evalprob_qcdfromtop_5m,evalprob_qcdfromqcd_2m,evalprob_topfromqcd_2m,evalprob_topfromtop_2m,evalprob_qcdfromtop_2m,path_to_plots,plot_title):

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

test_results_dir='/Users/humbertosmac/Documents/work/Transformers/Transformers_finke/test_results/'
plot_title='JetClass'
path_to_plots=test_results_dir+'/All_ttbar_qcd/'
os.makedirs(path_to_plots,exist_ok=True)
#5M

qcd_file_dir_5m=test_results_dir+'/zjettonunu_run_b_7/'
qcd_file_name_qcd_5m='results_test_eval_200k.npz'
qcd_file_name_top_5m='results_test_eval_other_200k.npz'


top_file_dir_5m=test_results_dir+'/ttbar_run_b_1/'
top_file_name_top_5m='results_test_eval_200k.npz'
top_file_name_qcd_5m='results_test_eval_other_200k.npz'
    







evalprob_qcdfromqcd_5m,evalprob_topfromqcd_5m=GetDataFromQCDT(qcd_file_dir_5m,qcd_file_name_qcd_5m,qcd_file_name_top_5m)
evalprob_topfromtop_5m,evalprob_qcdfromtop_5m=GetDataFromTopT(top_file_dir_5m,top_file_name_top_5m,top_file_name_qcd_5m)



###2M

qcd_file_dir_2m=test_results_dir+'/zjettonunu_run_b_6/'
qcd_file_name_qcd_2m='results_test_eval_200k.npz'
qcd_file_name_top_2m='results_test_eval_other_200k.npz'


top_file_dir_2m=test_results_dir+'/ttbar_run_b_2_5/'
top_file_name_top_2m='results_test_eval_200k.npz'
top_file_name_qcd_2m='results_test_eval_other_200k.npz'
    

evalprob_qcdfromqcd_2m,evalprob_topfromqcd_2m=GetDataFromQCDT(qcd_file_dir_2m,qcd_file_name_qcd_2m,qcd_file_name_top_2m)
evalprob_topfromtop_2m,evalprob_qcdfromtop_2m=GetDataFromTopT(top_file_dir_2m,top_file_name_top_2m,top_file_name_qcd_2m)


'''
plot_probs(evalprob_qcdfromqcd_5m,evalprob_topfromqcd_5m,evalprob_topfromtop_5m,evalprob_qcdfromtop_5m,evalprob_qcdfromqcd_2m,evalprob_topfromqcd_2m,evalprob_topfromtop_2m,evalprob_qcdfromtop_2m,path_to_plots,plot_title)
plot_contour(evalprob_qcdfromqcd_5m,evalprob_topfromtop_5m,evalprob_qcdfromqcd_2m,evalprob_topfromtop_2m,path_to_plots,plot_title)

'''

s_top_2m=ComputeLLR(evalprob_topfromtop_2m,evalprob_topfromqcd_2m,'top')
s_qcd_2m=ComputeLLR(evalprob_qcdfromtop_2m,evalprob_qcdfromqcd_2m,'qcd')

s_top_5m=ComputeLLR(evalprob_topfromtop_5m,evalprob_topfromqcd_5m,'top')
s_qcd_5m=ComputeLLR(evalprob_qcdfromtop_5m,evalprob_qcdfromqcd_5m,'qcd')


#PlotLLR(s_qcd_5m,s_top_5m,s_qcd_2m,s_top_2m,path_to_plots,plot_title)
ROCcurve(s_qcd_5m,s_top_5m,s_qcd_2m,s_top_2m,path_to_plots,plot_title)

exit()
plot_color(evalprob)
plot_contour(evalprob)
