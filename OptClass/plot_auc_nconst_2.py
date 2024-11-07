import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from data_eval_helpers import read_file,extract_value
from sklearn.metrics import roc_curve, roc_auc_score

from decimal import Decimal



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
    plt.savefig(path_to_plots+'/plot_LLR_test_1.pdf')
    plt.close()
    return


def ROCcurveLLR(s_qcd,s_top,path_to_plots,plot_title,color):
    
    fpr, tpr, _ = roc_curve(np.append(np.zeros(len(s_qcd)), np.ones(len(s_top))), np.append(s_qcd, s_top))
    #fpr, tpr, _ = roc_curve(np.append(np.zeros(len(s_top)), np.ones(len(s_qcd))), np.append(s_top, s_qcd))
    #fpr, tpr, _ = roc_curve(np.append(np.ones(len(s_qcd)), np.zeros(len(s_top))), np.append(s_qcd, s_top))
    print(roc_auc_score(np.append(np.zeros(len(s_qcd)), np.ones(len(s_top))), np.append(s_qcd, s_top)))
    auc_score=roc_auc_score(np.append(np.zeros(len(s_qcd)), np.ones(len(s_top))), np.append(s_qcd, s_top))

    plt.plot(tpr,1/fpr, label='LLR $N_{const}=$'+str(num_const)+'AUC='+str(truncate_float(auc_score,5)),linestyle='-',color=color)
    plt.ylim(1, 1e8)
    plt.xlim(0, 1)
    plt.xlabel(r"$\epsilon_{\rm{top}}$")
    plt.ylabel(r"$1 / \epsilon_{\rm{QCD}}$")
    plt.yscale('log')
    #plt.xscale('log')
   

    return


def truncate_float(float_number, decimal_places):
    multiplier = 10 ** decimal_places
    return int(float_number * multiplier) / multiplier

def plot_roc_curve(predictions,num_const,color):
    # Compute ROC curve and ROC area for each class
    

    
    fpr, tpr, _ = roc_curve(predictions['labels'], predictions['predictions'])
    #fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = roc_auc_score(predictions['labels'],predictions['predictions'])
    
    plt.plot(tpr,1/fpr, label='C $N_{const}=$'+str(num_const)+'. AUC='+str(truncate_float(roc_auc,5)),linestyle='--',color=color)
  

    return


def plot_probs(evalprob,path_to_plots,plot_title):

 plt.hist(evalprob['probs'],histtype='step',bins=30,density=True,color='blue')

 exp_logp=ExpectedProb(evalprob)

 plt.hist(exp_logp,histtype='step',bins=30,linestyle='--',density=True,color='blue')
 plt.title(plot_title,loc='left')
 plt.savefig(path_to_plots+'/plot_probs_test_1.pdf')
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
 plt.savefig(path_to_plots+'/plot_probs_test_1.pdf')
 plt.close()

 return

def GetPredictions(classifier_dir,num_const):

    predicitions_test = np.load(classifier_dir+'/predictions_test.npz')
    return predicitions_test


def plot_color(evalprob,path_to_plots,plot_title):

 plt.plot(evalprob['probs'],evalprob['n_const'],'.')
 plt.xlabel('n_{const}')
 plt.ylabel('log(p)')
 plt.title(plot_title,loc='left')
 plt.savefig(path_to_plots+'/plot_plot_test_1.pdf')
 plt.close()

 return
 
 def extract_value(var,lines):

    for line in lines:
        if 'lr_' in line:
            continue
        if var in line:
            line=line.replace(' ','')
            line=line.replace('\n','')

            value=line.split(var)[-1]

    return value
 
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
plot_title='Classifiers vs LLR'
#plot_title='JetClass test events'

qcd_file_dir=test_results_dir+'/ZJetsToNuNu_run_scan_10M_N1G96CW/'
top_file_dir=test_results_dir+'/TTBar_run_testwall_10M_11/'

dir_classifier_results='/Users/humbertosmac/Documents/work/Transformers/Optimal_Classifier/Classifier_results/TransformerClassifier/Classification_optclass_nconst/'

path_to_plots=dir_classifier_results

classif_dirs=os.listdir(dir_classifier_results)

num_const_list=[100,80,60,40,20,10,5,2]

color_list=['blue','red','green','purple','brown','orange','black','pink']
i=0
for num_const_l in num_const_list:

    for classif_dir in classif_dirs:
    
        if classif_dir=='plot_ROCcurve_nconst_trained.pdf':
            current_dir='nodir'
            continue
        arguments_file=read_file(dir_classifier_results+classif_dir+'/arguments.txt')
        num_const_file=int(extract_value('num_const',arguments_file))
        
        
        if num_const_l==num_const_file:
            num_const=num_const_l
            current_dir=dir_classifier_results+classif_dir
            continue
      
    if current_dir=='nodir':
        continue

    color=color_list[i]
    qcd_file_name_qcd='results_test_eval_optclass_testset_nsamples200000_num_const_'+str(num_const)+'.npz'
    qcd_file_name_top='results_test_eval_optclass_testset_other_nsamples200000_num_const_'+str(num_const)+'.npz'



    top_file_name_top='results_test_eval_optclass_testset_nsamples200000_num_const_'+str(num_const)+'.npz'
    top_file_name_qcd='results_test_eval_optclass_testset_other_nsamples200000_num_const_'+str(num_const)+'.npz'
    

    evalprob_qcdfromqcd,evalprob_topfromqcd=GetDataFromQCDT(qcd_file_dir,qcd_file_name_qcd,qcd_file_name_top)
    evalprob_topfromtop,evalprob_qcdfromtop=GetDataFromTopT(top_file_dir,top_file_name_top,top_file_name_qcd)

    s_top=ComputeLLR(evalprob_topfromtop,evalprob_topfromqcd,'qcd')
    s_qcd=ComputeLLR(evalprob_qcdfromtop,evalprob_qcdfromqcd,'qcd')

    ROCcurveLLR(s_qcd,s_top,path_to_plots,plot_title,color)




    
    predicitions_test=GetPredictions(current_dir,num_const)

    plot_roc_curve(predicitions_test,num_const,color)
    i=i+1


plt.ylim(1, 2e6)
plt.xlim(0, 1)
#plt.xlabel(r"$\epsilon_{\rm{top}}$")
#plt.ylabel(r"$1 / \epsilon_{\rm{QCD}}$")
plt.xlabel('TPR')
plt.ylabel('1/FPR')
plt.yscale('log')
#plt.xscale('log')
plt.legend(loc='upper right')
plt.title(plot_title,loc='left')
plt.savefig(path_to_plots+'/plot_ROCcurve_nconst_trained.pdf')
plt.close()



