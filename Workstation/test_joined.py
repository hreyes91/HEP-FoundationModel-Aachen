import pandas as pd
import matplotlib.pyplot as plt
from data_eval_helpers import make_continues,Make_Plots,LoadTrue,LoadSGenamples,GetHighLevel,Wasserstein_distance
import os
import numpy as np

'''
def readFrameCont(path,n_samples):
    tmp = pd.read_hdf(discrete_truedata_filename, key="discretized", stop=None)
    tmp=tmb.sample(n_samples)
    return tmp


bins_path_prefix='../preprocessing_bins/'
bin_tag='10M_TTBar_ZJetsToNuNu'
pt_bins = np.load(bins_path_prefix+'pt_bins_'+bin_tag+'.npy')
eta_bins = np.load(bins_path_prefix+'eta_bins_'+bin_tag+'.npy')
phi_bins = np.load(bins_path_prefix+'phi_bins_'+bin_tag+'.npy')

n_test_samples=100000
file_name_samples='/net/data_t2k/transformers-hep/JetClass/discretized/ALL_1M_val___10M_ALL_1M.h5'

jets,ptj,mj=LoadSGenamples(file_name_samples,pt_bins,eta_bins,phi_bins,n_test_samples)

discrete_truedata_filename='/net/data_t2k/transformers-hep/JetClass/discretized/ALL_1M_test___10M_ALL_1M.h5'

jets_true,ptj_true,mj_true=LoadTrue(discrete_truedata_filename,n_test_samples,pt_bins,eta_bins,phi_bins)


path_to_plots='test_plots_join_all_test_val_'+str(n_test_samples)
os.makedirs(path_to_plots,exist_ok=True)
Make_Plots(jets,pt_bins,eta_bins,phi_bins,mj,jets_true,ptj_true,mj_true,path_to_plots)
'''

def GetEvalDataJoined(file_dir,eval_qcd,eval_top):


    file_qcd=file_dir+'/'+eval_qcd
    evalprob_qcd = np.load(file_qcd)
    
    file_top=file_dir+'/'+eval_top
    evalprob_top = np.load(file_top)



    return evalprob_top['probs'],evalprob_qcd['probs']
    

def GetEvalDataTop(file):

    evalprob_top_best = np.load(file)

    return evalprob_top_best



def GetEvalDataQCD(file):

  
    evalprob_qcd_best = np.load(file)

    return evalprob_qcd_best
    
    
def plot_probs(evalprob_best,evalprob_joined,path_to_plots,tag,plot_title):

 plt.hist(evalprob_best,histtype='step',bins=50,density=True,color='blue',label='single')
 plt.hist(evalprob_joined,histtype='step',bins=50,density=True,color='blue',linestyle='--',label='joined')
 
 plt.xlabel('log(p)')
 plt.legend()
 plt.title(plot_title,loc='left')
 plt.savefig(path_to_plots+'/plot_probs_'+tag+'.png')
 plt.close()

 return


def EvalProbs(num_samples_test,num_const_test,model_path_curr,model_name,test_dataset,tag_foreval):

    
    command_eval='python ../evaluate_probabilities.py --fixed_samples --model '+str(model_path_curr)+'/'+str(model_name)+' --data '+str(test_dataset)+' --tag '+tag_foreval+' --num_const '+str(num_const_test)+' --num_events '+str(num_samples_test)
        
        
    os.system(command_eval)
    
    
    print(command_eval)
    return

def BayesFactor(evalprob,evalprob_true):


    LR_statistic =  (np.sum(evalprob) /np.sum(evalprob_true))
    
    
    return LR_statistic

def LLR2(evalprob,evalprob_true):

    N=int(np.shape(evalprob_true)[0])
    print(N)
    LR_statistic_2 =  np.abs(np.sum(evalprob-evalprob_true)/N)
    
    
    return LR_statistic_2


def read_file(file_name):

    f = open(file_name, "r")
    lines = f.readlines()

    return lines

def extract_value(var,lines):

    for line in lines:
        if 'num_events_val' in line:
            continue
        if var in line:
            line=line.replace(' ','')
            line=line.replace('\n','')

            value=line.split(var)[-1]
    
    return value

test_results_dir='/net/data_t2k/transformers-hep/JetClass/'




joined_file_dir=test_results_dir+'/TTBar_ZJetsToNuNu_models/'




model_name='model_best.pt'
num_const_test=100
num_samples_test=200000
tag_foreval='nconst_eval_nsamples'+str(num_samples_test)

test_dataset_top='/net/data_t2k/transformers-hep/JetClass/discretized/TTBar_test___10M_TTBar.h5'
model_path_curr=test_results_dir+'/TTBar_models/TTBar_run_testwall_10M_11/'
#EvalProbs(num_samples_test,num_const_test,model_path_curr,model_name,test_dataset_top,tag_foreval)

test_dataset_qcd='/net/data_t2k/transformers-hep/JetClass/discretized/ZJetsToNuNu_test___10M_ZJetsToNuNu.h5'
model_path_curr=test_results_dir+'//ZJetsToNuNu_models/ZJetsToNuNu_run_scan_10M_N1G96CW/'
#EvalProbs(num_samples_test,num_const_test,model_path_curr,model_name,test_dataset_qcd,tag_foreval)


qcd_file_name=test_results_dir+'ZJetsToNuNu_models/ZJetsToNuNu_run_scan_10M_N1G96CW/results_nconst_eval_nsamples'+str(num_samples_test)+'.npz'
evalprob_qcd_best=GetEvalDataQCD(qcd_file_name)['probs']

top_file_name=test_results_dir+'/TTBar_models/TTBar_run_testwall_10M_11/results_nconst_eval_nsamples'+str(num_samples_test)+'.npz'
evalprob_top_best=GetEvalDataTop(top_file_name)['probs']



joined_result_tag='TTBar_ZJetsToNuNu_run_test_joined_403030_'
joined_result_list=['0S1DG44','GGZNTEU','RTP0JN6']


tag_foreval_qcd='nconst_eval_qcd_nsamples'+str(num_samples_test)
tag_foreval_top='nconst_eval_top_nsamples'+str(num_samples_test)
for joined_result in joined_result_list:

    
    path=joined_file_dir+'/'+joined_result_tag+joined_result+'/'
    print(path)
    file_name=path+'/arguments.txt'
    lines=read_file(file_name)
    num_samples=extract_value('num_events',lines)
    
    
 
    model_path_curr=path
    #EvalProbs(num_samples_test,num_const_test,model_path_curr,model_name,test_dataset_qcd,tag_foreval_qcd)
    #EvalProbs(num_samples_test,num_const_test,model_path_curr,model_name,test_dataset_top,tag_foreval_top)
    
    
    eval_qcd='results_nconst_eval_qcd_nsamples'+str(num_samples_test)+'.npz'
    eval_top='results_nconst_eval_top_nsamples'+str(num_samples_test)+'.npz'
    evalprob_top,evalprob_qcd=GetEvalDataJoined(path,eval_qcd,eval_top)
    
    
    bayes_factor=BayesFactor(evalprob_top,evalprob_top_best)
    print(joined_result)
    print('top')
    print(bayes_factor)
    llr_2=LLR2(evalprob_top,evalprob_top_best)
    print(llr_2)
    
    
    plot_probs(evalprob_top_best,evalprob_top,path,'TTBar','TTBar -- LLR:'+str(llr_2)+' -- n_samples:'+str(num_samples))
    
    
    bayes_factor=BayesFactor(evalprob_qcd,evalprob_qcd_best)
    print('qcd')
    print(bayes_factor)

    llr_2=LLR2(evalprob_qcd,evalprob_qcd_best)
    print(llr_2)
    plot_probs(evalprob_qcd_best,evalprob_qcd,path,'QCD','QCD -- LLR:'+str(llr_2)+' -- n_samples:'+str(num_samples))
    
