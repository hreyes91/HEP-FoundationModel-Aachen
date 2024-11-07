import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
from data_eval_helpers import make_continues,Make_Plots,LoadTrue,LoadSGenamples,GetHighLevel,Wasserstein_distance
from scipy import stats


def GetDataEval(file_dir):
    file_name='results_test_eval_nsamples200000.npz'

    
    file=file_dir+'/'+file_name
    print(file)
    evalprob = np.load(file)

    return evalprob




def plot_probs(evalprob,num_const,color):
    print(evalprob['probs'])
    plt.hist(evalprob['probs'],histtype='step',bins=80,density=True,color=color,label=str(num_const))
    plt.legend(fontsize="large")

    return
 
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


def plot_multiplicity(path_to_plots,num_const,color):


    mask = jets[:, :, 0] != 0
    plt.hist(np.sum(mask, axis=1), bins=np.linspace(-0.5, 100.5, 102),color=color,histtype='step',density=True,linestyle='dashed',label=str(num_const))
    
    plt.xlabel('Multiplicity')
    plt.legend()
    plt.savefig(path_to_plots+'/plot_mul.png')




    return


def dict_to_frame_and_save(data_dict,path):

    frame=pd.DataFrame(data_dict)
    frame=frame.sort_values(by=['num_const'])
    frame.to_csv(path+'/results.txt',index=False)
    
    return frame


def LLR2(evalprob,evalprob_true):

    N=int(np.shape(evalprob_true)[0])
    print(N)
    LR_statistic_2 =  np.abs(np.sum(evalprob-evalprob_true)/N)
    
    
    return LR_statistic_2


mother_dir='/net/data_t2k/transformers-hep/JetClass/TTBar_models/test_const_dep/'
#results_tag='TTBar_run_scan_const_1M_'

results_list=os.listdir(mother_dir)
print(len(results_list))
colors=['blue','green','red', 'cyan','magenta','orange','black']

j=0
for k  in range(len(results_list)):
    result=results_list[k]
    color=colors[j]

    file_dir=mother_dir+'/'+result+'/'
    try:
        
        arguments_file=read_file(file_dir+'arguments.txt')
        num_const=extract_value('num_const',arguments_file)
        
        if num_const=='120':
            continue
        if num_const=='20':
            continue
        if num_const=='10':
            continue
        print(result)
        print(num_const)
        evalprob=GetDataEval(file_dir)
        print(evalprob)
        plot_probs(evalprob,num_const,color)
        j=j+1
    except:
        continue
    

plt.xlabel('log(p)')
plt.legend(fontsize="large")
plt.title('TTBar -- multiplicity')
plt.savefig(mother_dir+'plot_probs_test_1.png')
plt.close()


######## PLOT SAMPLES #######






bins_path_prefix='../preprocessing_bins/'
bin_tag='10M_TTBar'
pt_bins = np.load(bins_path_prefix+'pt_bins_'+bin_tag+'.npy')
eta_bins = np.load(bins_path_prefix+'eta_bins_'+bin_tag+'.npy')
phi_bins = np.load(bins_path_prefix+'phi_bins_'+bin_tag+'.npy')

n_test_samples=190000
discrete_truedata_filename='/net/data_t2k/transformers-hep/JetClass/discretized/TTBar_test___10M_TTBar.h5'
jets_true,ptj_true,mj_true=LoadTrue(discrete_truedata_filename,n_test_samples,pt_bins,eta_bins,phi_bins)
pt_true, eta_true,phi_true,mul_true=GetHighLevel(jets_true)

mask = jets_true[:, :, 0] != 0
plt.hist(np.sum(mask, axis=1), bins=np.linspace(-0.5, 100.5, 102),color='black',histtype='step',density=True,label='True')

j=0
for k  in range(len(results_list)):
    result=results_list[k]
    color=colors[j]

    

    file_dir=mother_dir+'/'+result+'/'
    try:
    
        arguments_file=read_file(file_dir+'arguments.txt')
        num_const=extract_value('num_const',arguments_file)
        if num_const=='10':
            continue
        file_name_samples=mother_dir+'/'+result+'/samples__nsamples200000_trunc_5000.h5'

        jets,ptj,mj=LoadSGenamples(file_name_samples,pt_bins,eta_bins,phi_bins,n_test_samples)
        print('jets')
        pt, eta,phi,mul=GetHighLevel(jets)
   
        print('high level')
        plot_multiplicity(mother_dir,num_const,color)
        print('plot')
        j=j+1
    except:
        continue
plt.title('TTBar -- multiplicity')
plt.close()


data_dict_result={'num_const':[],'w_distance_mul':[] ,'bayes_factor':[]}


model_100_file=mother_dir+'/TTBar_run_scan_const_1M_CIAH1GO/'
evalprob_100=GetDataEval(model_100_file)['probs']



for k  in range(len(results_list)):
    result=results_list[k]
    file_dir=mother_dir+'/'+result+'/'
    try:
    
        arguments_file=read_file(file_dir+'arguments.txt')
        num_const=extract_value('num_const',arguments_file)
        if num_const=='120':
            continue
        file_name_samples=mother_dir+'/'+result+'/samples__nsamples200000_trunc_5000.h5'
    
        jets,ptj,mj=LoadSGenamples(file_name_samples,pt_bins,eta_bins,phi_bins,n_test_samples)
        print('jets')
        pt, eta,phi,mul=GetHighLevel(jets)
        print('mul')
        print(mul)
        w_distance_mul=Wasserstein_distance(mul_true,mul)
        print(w_distance_mul)
        evalprob=GetDataEval(file_dir)['probs']
        print('eval prob')
        print(evalprob)
        print(evalprob_100)
        print()
        #LR_statistic =  (np.sum(evalprob) /np.sum(evalprob_100))
        LR_statistic =LLR2(evalprob,evalprob_100)
        print('LR')
        print(LR_statistic)
        data_dict_result.get('w_distance_mul').append(w_distance_mul)
        data_dict_result.get('num_const').append(int(num_const))
        data_dict_result.get('bayes_factor').append(LR_statistic)
    except:
        continue
    


frame=dict_to_frame_and_save(data_dict_result,mother_dir)
