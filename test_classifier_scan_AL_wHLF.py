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


    plt.plot(frame_all['num_events'],frame_all['auc_score'],color='blue')
    plt.plot(frame_all['num_events'],frame_all['auc_score'],'.',color='blue')
    plt.xlabel('$N_{train}$')
    plt.ylabel('auc score')
    plt.xscale('log')
    plt.savefig(mother_dir+'/auc_events_optclass.png')
    

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


def read_file(file_name):

    f = open(file_name, "r")
    lines = f.readlines()

    return lines



data_path_1='//net/data_ttk/hreyes/OneBin/TTBar_test___1Mfromeach_403030.h5'
data_path_2='///net/data_ttk/hreyes/OneBin/ZJetsToNuNu_test___1Mfromeach_403030.h5'
mother_dir='//net/data_ttk/hreyes/JetClass/Classification_CLS/top_vs_qcd//'
#tag='top_vs_qcd_jetclass_classifier_test_1_'
num_events=200000
#num_const=128
model_dirs=os.listdir(mother_dir)

dict_auc={'num_events':[],'auc_score':[]}



for model_dir in model_dirs:
    if 'nevents1000' not in model_dir:
        continue
    
    file_name=mother_dir+model_dir+'/arguments.txt'
    val='num_const'
    lines=read_file(file_name)
    num_const=extract_value(val,lines)
    print('hello')
    print(mother_dir+model_dir)
    #print(os.listdir(mother_dir+model_dir))
    command='python test_classifier_cls.py --data_path_1 '+data_path_1+' --data_path_2 '+data_path_2+' --model_dir '+ mother_dir+model_dir+'  --num_events '+str(num_events)+' --num_const '+str(num_const)+' --pred_name '+str('predictions_test.npz')
    print(command)
    try:
        os.system(command)
        print('there')
        arguments_file=read_file(mother_dir+model_dir+'/arguments.txt')
        num_events_train=extract_value('num_events',arguments_file)
        print(num_events_train)
        auc=readauc(mother_dir+model_dir)
        print(auc)
        dict_auc.get('num_events').append(num_events_train)
        dict_auc.get('auc_score').append(auc)
        print('done '+model_dir)
    except:
        continue





frame_all=pd.DataFrame(dict_auc)
frame_all=frame_all.sort_values(by=['num_events'])
frame_all.to_csv(mother_dir+'/frame_auc.txt',index=False)
PlotAUCwN(frame_all,mother_dir)


