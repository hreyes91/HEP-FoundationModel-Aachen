import os
import pandas as pd
import matplotlib.pyplot as plt

def readauc(dir):
    
    frame=pd.read_csv(dir+'/auc_optclass.txt')
    
    auc_score=float(frame['auc_score'])
    
    return auc_score


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


def PlotAUCwN(frame_all,model_dir):


    plt.plot(frame_all['num_const'],frame_all['auc_score'],color='blue')
    plt.plot(frame_all['num_const'],frame_all['auc_score'],'.',color='blue')
    plt.xlabel('$N_{train}$')
    plt.ylabel('auc score')
    plt.xscale('log')
    plt.savefig(mother_dir+'/auc_events_optclass.png')
    

    return
    

data_path_1='/net/data_t2k/transformers-hep/JetClass//TTBar_models//TTBar_run_testwall_10M_11/samples_samples_nsamples200000_trunc_5000.h5'
data_path_2='/net/data_t2k/transformers-hep/JetClass//ZJetsToNuNu_models//ZJetsToNuNu_run_scan_10M_N1G96CW/samples__nsamples200000_trunc_5000.h5'
mother_dir='/net/data_t2k/transformers-hep/JetClass/Classification_optclass/'
#tag='top_vs_qcd_jetclass_classifier_test_1_'
num_events=200000

model_dirs=os.listdir(mother_dir)

dict_auc={'num_const':[],'auc_score':[],'file':[]}


for model_dir in model_dirs:
    if 'nconst_take2' not in model_dir:
        continue
    try:
        arguments_file=read_file(mother_dir+model_dir+'/arguments.txt')
        num_const=extract_value('num_const',arguments_file)
        print(num_const)
        
        command='python test_classifier.py --data_path_1 '+data_path_1+' --data_path_2 '+data_path_2+' --model_dir '+ mother_dir+model_dir+'  --num_events '+str(num_events)+'  --num_const '+str(num_const)
        os.system(command)
        print('hello')
        
        auc=readauc(mother_dir+model_dir)
        print(auc)
        dict_auc.get('num_const').append(num_const)
        dict_auc.get('auc_score').append(auc)
        dict_auc.get('file').append(model_dir)
        print('done '+model_dir)
    except:
        continue





frame_all=pd.DataFrame(dict_auc)
frame_all=frame_all.sort_values(by=['num_const'])
frame_all.to_csv(mother_dir+'/frame_auc_nconst_2.txt',index=False)
PlotAUCwN(frame_all,mother_dir)


