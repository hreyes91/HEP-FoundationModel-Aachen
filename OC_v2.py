import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn import metrics

#from data_eval_helpers import LoadTrue,make_continues,GetHighLevel,Make_Plots,Load,MakeIndividualPlots

def GetPredictions(path_to_prediction ):

    #predicitions_test = np.load(classifier_dir+'/predictions_test_'+str(num_const)+'.npz')
    
    predicitions = np.load(path_to_prediction,allow_pickle=True)
    
    return predicitions



def GetROC(LLR_TOP,LLR_QCD):

    fpr, tpr, thresholds = metrics.roc_curve(np.append(np.zeros(len(LLR_QCD)), np.ones(len(LLR_TOP))), np.append(LLR_QCD, LLR_TOP))
    auc=metrics.auc(fpr, tpr)
    return fpr, tpr, auc

def PlotLLR(fpr,tpr,auc,epoch):

    plt.plot(tpr,1/fpr,'-',label='epoch '+str(epoch)+'='+str(auc))


    return



def ComputeLLR(evalprobT,evalprobF,type):

    #evalprobT_P = np.where(evalprobT['probs'] == -np.inf, -1e-40, evalprobT['probs'])[:]
    #evalprobF_P = np.where(evalprobF['probs'] == -np.inf, -1e-40, evalprobF['probs'])[:]
    
    
    s=evalprobT-evalprobF
    #s=np.exp(evalprobT['probs'])/np.exp(evalprobF['probs'])
    return s


def FinishPlot():

    plt.title('LLR evolution')
    plt.legend()
    plt.xlabel('tagging efficiency')
    plt.ylabel('rejection efficiency')
    plt.savefig('LLRevolution.png')
    plt.close()



    return

    
main_path='/net/data_ttk/hreyes/FLJT/ResultsV2/'

epoch_list=[0,1]

for epoch in epoch_list:

    path_to_TOP_topT=main_path+'/Tests_TOP_RUN1/results_eval_samples_epoch_'+str(epoch)+'.npz'
    path_to_QCD_qcdT=main_path+'/Tests_QCD_RUN1/results_eval_samples_epoch_'+str(epoch)+'.npz'

    path_to_QCD_topT=main_path+'/Tests_TOP_RUN1/results_eval_samples_other_epoch_'+str(epoch)+'.npz'
    path_to_TOP_qcdT=main_path+'/Tests_QCD_RUN1/results_eval_samples_other_epoch_'+str(epoch)+'.npz'

    probs_TOP_topT=GetPredictions(path_to_TOP_topT)['probs'][:99]
    print(np.shape(probs_TOP_topT))
    probs_QCD_qcdT=GetPredictions(path_to_QCD_qcdT)['probs'][:99]

    probs_TOP_qcdT=GetPredictions(path_to_TOP_qcdT)['probs'][:99]
    probs_QCD_topT=GetPredictions(path_to_QCD_topT)['probs'][:99]
    

    LLR_TOP=ComputeLLR(probs_TOP_topT,probs_TOP_qcdT,'_')
    LLR_QCD=ComputeLLR(probs_QCD_topT,probs_QCD_qcdT,'_')

    fpr, tpr, auc=GetROC(LLR_TOP,LLR_QCD)

    PlotLLR(fpr,tpr,auc,epoch)


FinishPlot()
