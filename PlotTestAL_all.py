import os
import numpy as np
import h5py
import pandas as pd
import matplotlib.pyplot as plt
import random
import string
from sklearn.metrics import roc_curve, roc_auc_score,accuracy_score





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


def GetPredictions(model_dir):

    predicitions_test = np.load(model_dir+'/predictions_test.npz')
    
    
    return predicitions_test
    





def ComputeR(tpr,fpr,r_tresh):
    # Compute ROC curve and ROC area for each class
    

    results= np.stack((tpr, 1/fpr), -1)
    
    results_pandas=pd.DataFrame(results,columns=['tpr','1/fpr'])
    
    print(results)
    print(results_pandas)
    threshold = r_tresh

    # Select rows where 'Column 1' values are above the threshold
    filtered_df = results_pandas[results_pandas['tpr'] < threshold]
    print(filtered_df)
    r_val=float(filtered_df.sort_values(by=['tpr'],ascending=False).reset_index()[['1/fpr']].iloc[0])

    return r_val


def Accuracy(predictions,labels):

    fixed_pred=predictions
    fixed_pred[fixed_pred >= 0.5] = 1
    fixed_pred[fixed_pred < 0.5] = 0

    acc=accuracy_score(labels, fixed_pred,normalize=True)
    
    return acc

def truncate_float(float_number, decimal_places):
    multiplier = 10 ** decimal_places
    return int(float_number * multiplier) / multiplier

'''
def PlotRocCurve(fpr, tpr,num_events_train,r_tresh,r_val):

    plt.plot(tpr,1/fpr, label='$sig injec=$'+str(num_events_train)+'_R'+str(r_tresh*100)+'='+str(truncate_float(r_val,4)),linestyle='--') #color=color)
    plt.ylim(1, 1e8)
    plt.xlim(0, 1)
    plt.xlabel(r"$pr$")
    plt.ylabel(r"$1 / fpr$")
    plt.yscale('log')
    
    
    return
'''


def PlotRocCurve(fpr, tpr,num_events_train,auc):

    plt.plot(tpr,1/fpr, label='$sig injec=$'+str(num_events_train)+'AUC='+str(truncate_float(auc,4)),linestyle='--') #color=color)
    plt.ylim(1, 1e8)
    plt.xlim(0, 1)
    plt.xlabel(r"$pr$")
    plt.ylabel(r"$1 / fpr$")
    plt.yscale('log')
    
    
    return


def TestMetrics(predictions,r_tresh):

    

    labels=predictions['labels']
    predictions=predictions['predictions']

    fpr, tpr, _ = roc_curve(labels, predictions)
    
    
    
    auc_score=roc_auc_score(labels, predictions)
    
    
    r_tresh=r_tresh
    #r_val=ComputeR(tpr,fpr,r_tresh)
    r_val=0
    
    acc=Accuracy(predictions,labels)
    
    
    sic_values = np.where(fpr > 0, tpr / np.sqrt(fpr), 0)

    # Find maximum SIC and corresponding threshold
    max_sic = np.max(sic_values)

    return auc_score,r_val,acc,max_sic,fpr,tpr




def TestResults(dict_auc,model_dir,r_tresh):


    arguments_file=read_file(model_dir+'/arguments.txt')
    num_events_train=extract_value('sig',arguments_file)
    
    num_events_train=int(num_events_train.split('/')[-1].split('-')[-1].split('.')[0])
    
    print(num_events_train)
    
    
    predictions=GetPredictions(model_dir)
    auc_score,r_val,acc,max_sic,fpr,tpr=TestMetrics(predictions,r_tresh)


    #PlotRocCurve(fpr, tpr,num_events_train,auc_score)

    dict_auc.get('sig').append(num_events_train)
    dict_auc.get('auc').append(auc_score)
    #dict_auc.get('r_'+str(r_tresh)).append(r_val)
    dict_auc.get('acc').append(acc)
    dict_auc.get('max_sic').append(max_sic)
  

    return
    
    
def PlotAUCs(auc_frame,path_to_plots,plot_title):

    plt.plot(auc_frame['sig'],auc_frame['auc'],'.',linestyle='-')
    plt.xlabel('signal injection')
    plt.ylabel('AUC')
    plt.title(plot_title)
    plt.savefig(path_to_plots+'auc.png')
    plt.close()
    
    
    return


def PlotMaxSICs(auc_frame,path_to_plots,plot_title):

    plt.plot(auc_frame['sig'],auc_frame['max_sic'],'.',linestyle='-')
    plt.xlabel('signal injection')
    plt.ylabel('max(SIC)')
    plt.title(plot_title)
    plt.savefig(path_to_plots+'max_sic.png')
    plt.close()
    
    
    return



#main_dir_discrete='/net/data_ttk/hreyes/LHCO/LHCO_discrete/'
main_dir_discrete='LHCO_discrete/'


data_path_1=main_dir_discrete+'discrete_1Mfromeach_403030_bg-N100-SR-Test.h5'
data_path_2=main_dir_discrete+'discrete_1Mfromeach_403030_sn-N100-SR-Test.h5'
r_tresh=.5
#dict_auc={'result':[],'sig':[],'auc':[],'r_'+str(r_tresh):[],'acc':[],'max_sic':[]}


main_result_dir='/Users/humbertosmac/Documents/work/Foundation_Model/AnomalyDetection/Results/Supervised/Classification_supervised_imbalanced/'


model_types={'scratch LL+HLF':'Classification_AL_supervised_scratch_wHLF_test_datav2_imbalanced_1/',
              'finetuned LL+HLF':'Classification_AL_supervised_finetune_wHLF_test_datav2_imbalanced_1/',
              'scratch LL':'Classification_AL_supervised_scratch_test_datav2_imbalanced_1/',
              'finetuned LL':'Classification_ALsupervised_finetune_AVGpoolhead_test_datav2_imbalanced_1/'}

epochs_list=[1,60]
for num_epochs in epoch_list:
    for model_type in model_types.keys():

        dict_auc={'result':[],'sig':[],'auc':[],'acc':[],'max_sic':[]}
        results_dirs=os.listdir(main_result_dir+model_types.get(model_type))

        for result in results_dirs:

                if 'LHCO_test_AL' not in result:
                    continue
                try:
                    model_dir=main_result_dir+model_types.get(model_type)+'/'+result
                  
                    
                    TestResults(dict_auc,model_dir,r_tresh)
                    dict_auc.get('result').append(result)
          
                    #except:
                    #    continue
                except:
                    continue

        '''
        plot_title='fine-tuned'
        plt.legend(loc='upper right')
        plt.title(plot_title,loc='left')
        
        plt.savefig(path_to_plots+'/ROC_curves_'+model_type+'.pdf')
        plt.close()
        '''

        auc_frame=pd.DataFrame(dict_auc)

        auc_frame=auc_frame.sort_values(by=['sig'])


        
        path_to_plots=main_result_dir+model_types.get(model_type)
        auc_frame.to_csv(path_to_plots+'results.txt',index=False)


        plot_title='LHCO-AachenT-QCDJetClass backbone-'+model_type
        #PlotAUCs(auc_frame,path_to_plots,plot_title)

        #PlotMaxSICs(auc_frame,path_to_plots,plot_title)

        print(model_type)
        plt.plot(auc_frame['sig'],auc_frame['auc'],'.',linestyle='-',label=model_type)
        


    plot_title='AUC'
    plt.xlabel('signal injection')
    plt.ylabel('m')
    plt.legend()
    plt.title(plot_title)
    plt.savefig(main_result_dir+'maxsic_all.png')
    plt.close()
