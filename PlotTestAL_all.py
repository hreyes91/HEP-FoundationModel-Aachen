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

def GetPredictionsLast(model_dir):

    predicitions_test_last = np.load(model_dir+'/predictions_test_last.npz')
    
    
    return predicitions_test_last

def GetPredictionsBestTrain(model_dir):

    predicitions_test_last = np.load(model_dir+'/predictions_test_best_train.npz')
    
    
    return predicitions_test_last

def GetPredictionsPerEpoch(model_dir,epoch):

    predictions_epoch = np.load(model_dir+'/predictions_test_epoch_'+str(epoch)+'.npz')
    
    
    return predictions_epoch
    





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
    print('Test metrocs')
    print(predictions)

    labels=predictions['labels']
    predictions=predictions['predictions']
    print(predictions)
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

    print('test results')
    arguments_file=read_file(model_dir+'/arguments.txt')
    num_events_train=extract_value('sig',arguments_file)
    
    num_events_train=int(num_events_train.split('/')[-1].split('-')[-1].split('.')[0])
    
    print(num_events_train)
    print('got num events train')
    
    predictions=GetPredictions(model_dir)
    predictions_last=GetPredictionsLast(model_dir)
    predictions_best_train=GetPredictionsBestTrain(model_dir)

    print('got predictions')
    auc_score,r_val,acc,max_sic,fpr,tpr=TestMetrics(predictions,r_tresh)

    print(auc_score)
    auc_score_last,r_val_last,acc_last,max_sic_last,fpr_last,tpr_last=TestMetrics(predictions_last,r_tresh)
    #PlotRocCurve(fpr, tpr,num_events_train,auc_score)

    auc_score_best_train,r_val_best_train,acc_best_train,max_sic_best_train,fpr_best_train,tpr_best_train=TestMetrics(predictions_best_train,r_tresh)

    dict_auc.get('sig').append(num_events_train)
    dict_auc.get('auc').append(auc_score)
    #dict_auc.get('r_'+str(r_tresh)).append(r_val)
    dict_auc.get('acc').append(acc)
    dict_auc.get('max_sic').append(max_sic)


    dict_auc.get('auc_last').append(auc_score_last)
    #dict_auc.get('r_'+str(r_tresh)).append(r_val)
    dict_auc.get('acc_last').append(acc_last)
    dict_auc.get('max_sic_last').append(max_sic_last)
  

    return num_events_train,auc_score,max_sic,auc_score_last,max_sic_last,fpr,tpr,fpr_last,tpr_last,auc_score_best_train,max_sic_best_train,fpr_best_train,tpr_best_train
    
    
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

def TestAgain(model_dir,model_name_test='best'):

    main_dir_discrete='/net/data_ttk/hreyes/LHCO/LHCO_discrete/'
    data_path_1=main_dir_discrete+'discrete_1Mfromeach_403030_bg-N100-SR-Test.h5'
    data_path_2=main_dir_discrete+'discrete_1Mfromeach_403030_sn-N100-SR-Test.h5'
    num_events_test=50000
    num_const=100

    test_command='python test_classifier_AL.py --data_path_1 '+data_path_1+' --data_path_2 '+data_path_2+' --model_dir '+ model_dir +'  --num_events '+str(num_events_test)+' --num_const '+str(num_const)+' --pred_name '+str('predictions_test.npz')+' --model_name '+str(model_name_test)
    os.system(test_command)

    return

def PlotPerEpoch(model_dir,frame_per_epoch,num_signal):

    plt.plot(frame_per_epoch['epoch'],frame_per_epoch['auc'])
    plt.xlabel('epoch')
    plt.ylabel('auc')
    plt.title('signal injection - '+str(num_signal))
    plt.savefig(model_dir+'/AUC_perepoch_'+str(num_signal)+'.png')
    plt.close()

    plt.plot(frame_per_epoch['epoch'],frame_per_epoch['max_sic'])
    plt.xlabel('epoch')
    plt.ylabel('max SIC')
    plt.title('signal injection - '+str(num_signal))
    plt.savefig(model_dir+'/max_sic_perepoch_'+str(num_signal)+'.png')
    plt.close()

    return

def TestPerEpoch(model_dir):

    lines=read_file(model_dir+'/arguments.txt')
    num_epochs=int(extract_value('num_epochs',lines))

    num_signal=int(extract_value('sig',lines).split('-')[-1].split('.')[0])
    print(num_signal)
    
    print(num_epochs)

    per_epoch_dict={'epoch':[],'auc':[],'max_sic':[],'acc':[]}
    for epoch in range(num_epochs):

        predictions_epoch=GetPredictionsPerEpoch(model_dir,epoch)
        print('hello')
        print(predictions_epoch)
        auc_score_epoch,r_val_epoch,acc_epoch,max_sic_epoch,fpr_epoch,tpr_epoch=TestMetrics(predictions_epoch,.5)
        print(auc_score_epoch)  
        per_epoch_dict.get('epoch').append(epoch)
        per_epoch_dict.get('auc').append(auc_score_epoch)
        per_epoch_dict.get('max_sic').append(max_sic_epoch)
        per_epoch_dict.get('acc').append(acc_epoch)
        print('whatup')
    print(per_epoch_dict)
  
    frame_per_epoch=pd.DataFrame(per_epoch_dict)

    PlotPerEpoch(model_dir,frame_per_epoch,num_signal)

    return num_signal

#main_dir_discrete='/net/data_ttk/hreyes/LHCO/LHCO_discrete/'
main_dir_discrete='LHCO_discrete/'
print('hello')


data_path_1=main_dir_discrete+'discrete_1Mfromeach_403030_bg-N100-SR-Test.h5'
data_path_2=main_dir_discrete+'discrete_1Mfromeach_403030_sn-N100-SR-Test.h5'
r_tresh=.5
#dict_auc={'result':[],'sig':[],'auc':[],'r_'+str(r_tresh):[],'acc':[],'max_sic':[]}


#main_result_dir='//net/data_ttk/hreyes/LHCO/IdealClassification/RUN10/'
#out_result_dir='./AnomalyDetectionResults/IdealizedClasifitaction/RUN10/'
#os.makedirs(out_result_dir, exist_ok=True)


model_types={#'scratch LL+HLF':'Classification_AL_scratch_wHLF_test_datav2_10/',
             # 'finetuned LL+HLF':'Classification_AL_finetune_wHLF_test_datav2_5/',
              'scratch LL':'Classification_AL_scratch_test_datav2/',
             # 'finetuned LL':'Classification_AL_finetune_AVGpoolhead_test_datav2_5/'
              
              }
all_runs={'RUN':[],'auc_best':[],'max_sic_best':[],'auc_last':[],'max_sic_last':[],'auc_best_train':[],'max_sic_best_train':[]}
rocs={'RUN':[],'fpr_best':[],'tpr_best':[],'fpr_last':[],'tpr_last':[]}
test_again='False'
first_run=9284
last_run=9285
for run in range(first_run,last_run):

    main_result_dir='//net/data_ttk/hreyes/LHCO/IdealClassification/RUN'+str(run)+'/'
    out_result_dir='./AnomalyDetectionResults/IdealizedClasifitaction/RUN'+str(run)+'/'
    os.makedirs(out_result_dir, exist_ok=True)
    for model_type in model_types.keys():

        dict_auc={'result':[],'sig':[],'auc':[],'acc':[],'max_sic':[],'auc_last':[],'acc_last':[],'max_sic_last':[]}
        results_dirs=os.listdir(main_result_dir+model_types.get(model_type))

        for result in results_dirs:

                if 'LHCO_CLS' not in result:
                    continue
                    
                try:
                    model_dir=main_result_dir+model_types.get(model_type)+'/'+result
                    print(model_dir)
                    if test_again=='True':
                        TestAgain(model_dir)

                    num_signal=TestPerEpoch(model_dir)
                    num_events_train,auc_best,max_sic_best,auc_last,max_sic_last,fpr_best,tpr_best,fpr_last,tpr_last,auc_score_best_train,max_sic_best_train,fpr_best_train,tpr_best_train=TestResults(dict_auc,model_dir,r_tresh)
                    os.system('cp '+model_dir+'/arguments.txt '+out_result_dir+'/arguments_'+str(num_events_train)+'.txt')
                    os.system('cp '+model_dir+'/auc.txt '+out_result_dir+'/auc_'+str(num_events_train)+'.txt')
                    os.system('cp '+model_dir+'/history.pdf '+out_result_dir+'/history_'+str(num_events_train)+'.pdf')

                    os.system('cp '+model_dir+'/max_sic_perepoch_'+str(num_signal)+'.png '+out_result_dir+'/max_sic_perepoch_'+str(num_signal)+'.png')
                    os.system('cp '+model_dir+'/AUC_perepoch_'+str(num_signal)+'.png '+out_result_dir+'/AUC_perepoch_'+str(num_signal)+'.png')
                    
                    dict_auc.get('result').append(result)

                    if num_events_train==1000:
                        all_runs.get('RUN').append(run)
                        all_runs.get('auc_best').append(auc_best)
                        all_runs.get('max_sic_best').append(max_sic_best)
                        all_runs.get('auc_last').append(auc_last)
                        all_runs.get('max_sic_last').append(max_sic_last)

                        all_runs.get('auc_best_train').append(auc_score_best_train)
                        all_runs.get('max_sic_best_train').append(max_sic_best_train)


                        rocs.get('RUN').append(run)
                        rocs.get('fpr_best').append(fpr_best)
                        rocs.get('tpr_best').append(tpr_best)
                        rocs.get('fpr_last').append(fpr_last)
                        rocs.get('tpr_last').append(tpr_last)
          
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
        auc_frame.to_csv(out_result_dir+'/results.txt',index=False)


        plot_title='LHCO-AachenT-QCDJetClass backbone-'+model_type
        #PlotAUCs(auc_frame,path_to_plots,plot_title)

        #PlotMaxSICs(auc_frame,path_to_plots,plot_title)

        print(model_type)
        plt.plot(auc_frame['sig'],auc_frame['auc'],'.',linestyle='-',label=model_type)
        


        plot_title='AUC best'
        plt.xlabel('signal injection')
        plt.ylabel('auc')
        plt.legend()
        plt.title(plot_title)
        plt.savefig(out_result_dir+'/auc_all_best.png')
        plt.close()

        plot_title='AUC last'
        plt.plot(auc_frame['sig'],auc_frame['auc_last'],'.',linestyle='-',label=model_type)
        plt.xlabel('signal injection')
        plt.ylabel('auc last')
        plt.legend()
        plt.title(plot_title)
        plt.savefig(out_result_dir+'/auc_all_last.png')
        plt.close()



all_runs_frame=pd.DataFrame(all_runs)
all_runs_frame.to_csv('./AnomalyDetectionResults/IdealizedClasifitaction/results_RUNS_'+str(first_run)+'_'+str(last_run)+'.txt',index=False)
rocs_frame=pd.DataFrame(rocs)

top_runs=all_runs_frame.nlargest(8, 'max_sic_last')

print(top_runs)

def PlotTopAUC(fpr,tpr,run,auc,max_sic):
    plt.plot(tpr,1/fpr, label='RUN='+str(run)+' AUC='+str(truncate_float(auc,3))+' maxSIC='+str(truncate_float(max_sic,3)),linestyle='--') #color=color)
    plt.ylim(1, 1e5)
    plt.xlim(0, 1)
    plt.xlabel(r"$ tpr$")
    plt.ylabel(r"$1 / fpr$")
    plt.yscale('log')





    return


for run in top_runs['RUN']:

    main_result_dir='//net/data_ttk/hreyes/LHCO/IdealClassification/RUN'+str(run)+'/'
    results_dirs=os.listdir(main_result_dir+model_types.get(model_type))
    for result in results_dirs:

        if 'LHCO_CLS' not in result:
            continue
                    
        try:
            model_dir=main_result_dir+model_types.get(model_type)+'/'+result
            num_events_train,auc_best,max_sic_best,auc_last,max_sic_last,fpr_best,tpr_best,fpr_last,tpr_last,auc_score_best_train,max_sic_best_train,fpr_best_train,tpr_best_train=TestResults(dict_auc,model_dir,r_tresh)
            PlotTopAUC(fpr_last,tpr_last,run,auc_last,max_sic_last)
        
        except:
            continue
    



#plt.plot([0, 1], [0, 1], linestyle='--', color='black')
print(top_runs)
plot_title='CLS head - signal injection = 1000 - top 5 (SIC  last)'
plt.legend()
plt.title(plot_title)
plt.savefig('./AnomalyDetectionResults/IdealizedClasifitaction/roc_curve_RUNS_'+str(first_run)+'_'+str(last_run)+'_SIC_last.png')
plt.close()
