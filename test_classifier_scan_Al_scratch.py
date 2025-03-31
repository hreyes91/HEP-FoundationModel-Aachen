
import os
import numpy as np
import h5py
import pandas as pd
import matplotlib.pyplot as plt
import random
import string
from sklearn.metrics import roc_curve, roc_auc_score,accuracy_score

def random_string():
    # initializing size of string
    N = 7
 
    # using random.choices()
    # generating random strings
    res = ''.join(random.choices(string.ascii_uppercase +
                             string.digits, k=N))
    # print result
    print("The generated random string : " + str(res))
    return str(res)
    
    
    
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
    


def GetPredictions(model_dir):

    predicitions_test = np.load(model_dir+'/predictions_test_best.npz')
    
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



def TestMetrics(predictions,r_tresh):

    

    labels=predictions['labels']
    predictions=predictions['predictions']

    fpr, tpr, _ = roc_curve(labels, predictions)
    
    
    auc_score=roc_auc_score(labels, predictions)
    
    
    r_tresh=r_tresh
    r_val=ComputeR(tpr,fpr,r_tresh)
    
    
    acc=Accuracy(predictions,labels)

    return auc_score,r_val,acc




def TestResults(dict_auc,model_dir,r_tresh):


    arguments_file=read_file(model_dir+'/arguments.txt')
    num_events_train=extract_value('sig',arguments_file)
    print(num_events_train)


    predictions=GetPredictions(model_dir)
    auc_score,r_val,acc=TestMetrics(predictions,r_tresh)

    dict_auc.get('sig').append(num_events_train)
    dict_auc.get('auc').append(auc_score)
    dict_auc.get('r_'+str(r_tresh)).append(r_val)
    dict_auc.get('acc').append(acc)
  

    return



main_dir_discrete='/net/data_ttk/hreyes/LHCO/LHCO_discrete/'



#test_data
data_path_1=main_dir_discrete+'discrete_1Mfromeach_403030_bg-N100-SR-Test.h5'
data_path_2=main_dir_discrete+'discrete_1Mfromeach_403030_sn-N100-SR-Test.h5'
num_events_test=50000
r_tresh=.3
dict_auc={'suffix':[],'sig':[],'auc':[],'r_'+str(r_tresh):[],'acc':[]}


tag_of_train='LHCO_test_AL_1'
log_dir='/net/data_ttk/hreyes/LHCO/IdealClassification/RUN4/Classification_AL_finetune_AVGpoolhead_test_datav2_5//'
model_name='model_best.pt'
model_name_test='best'
model_path_in='/net/data_ttk/hreyes/JetClass/OptClass/ZJetsToNuNu_models/ZJetsToNuNu_run_test__part_pt_1Mfromeach_403030_test_2_BU2IWA1/'

num_const=100
models_dirs=os.listdir(log_dir)

for model_dir in models_dirs:

                                                    
                                                    
                                                    
        model_dir=log_dir+model_dir
                                                    
        test_command='python test_classifier_AL.py --data_path_1 '+data_path_1+' --data_path_2 '+data_path_2+' --model_dir '+ model_dir +'  --num_events '+str(num_events_test)+' --num_const '+str(num_const)+' --pred_name '+str('predictions_test_best.npz')+' --model_name '+str(model_name_test)
                                                    
        os.system(test_command)
        TestResults(dict_auc,model_dir,r_tresh)
        name_sufix=model_dir.split('_')[-1]
        dict_auc.get('suffix').append(name_sufix)

