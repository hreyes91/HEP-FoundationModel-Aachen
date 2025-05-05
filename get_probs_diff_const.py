import numpy as np
import data_eval_helpers2 as deh
import evaluate_probabilities as probs
import os
from tqdm import tqdm
from tqdm import trange
import plot_llr as pl

n_const=[5,10,20,30,40,50,60,128]
data=['TTBar','HToGG','WToQQ','ZToQQ']
data_name=['top','h','w','z']
eval_data=False
eval_qcd=False

n_name='200k'

if eval_qcd==True:
    #for a in tqdm(n_const):
    a=40
    os.system(f'CUDA_VISIBLE_DEVICES=3 python evaluate_probabilities.py --model /net/data_ttk/koller/model_data/model_data_ZToNuNu_10M_train_ev_30_ep/model_best.pt  --data /net/data_ttk/koller/model_data/model_data_ZToNuNu_10M_train_ev_30_ep/samples_test_10M_k5000.h5 --tag ZToNuNu_{a}_const_{n_name}_ev --num_const {a}  --num_events 200000 --fixed_samples')

for j in range(len(data)):
    
    name=data[j]
    name_s=data_name[j]
    print(name)
    s_data_list=np.empty((len(n_const),200000))
    s_qcd_list=np.empty((len(n_const),200000))
    for k in trange(len(n_const)):
        n=n_const[k]
        print(n)
        if eval_data==True:
            if name=='TTBar':
                os.system(f'CUDA_VISIBLE_DEVICES=3 python evaluate_probabilities.py --model /net/data_ttk/koller/model_data/model_data_{name}_10M_train_ev_30_ep/model_best.pt  --data /net/data_ttk/koller/model_data/model_data_{name}_10M_train_ev_30_ep/samples_test_1M_k5000.h5 --tag {name}_{n}_const_{n_name}_ev --num_const {n}  --num_events 200000 --fixed_samples')
                os.system(f'CUDA_VISIBLE_DEVICES=3 python evaluate_probabilities.py --model /net/data_ttk/koller/model_data/model_data_ZToNuNu_10M_train_ev_30_ep/model_best.pt  --data /net/data_ttk/koller/model_data/model_data_{name}_10M_train_ev_30_ep/samples_test_1M_k5000.h5 --tag {name}_{n}_const_{n_name}_ev --num_const {n}  --num_events 200000 --fixed_samples')
            else:
                os.system(f'CUDA_VISIBLE_DEVICES=3 python evaluate_probabilities.py --model /net/data_ttk/koller/model_data/model_data_{name}_10M_train_ev_30_ep/model_best.pt  --data /net/data_ttk/koller/model_data/model_data_{name}_10M_train_ev_30_ep/samples_test_10M_k5000.h5 --tag {name}_{n}_const_{n_name}_ev --num_const {n}  --num_events 200000 --fixed_samples')
                os.system(f'CUDA_VISIBLE_DEVICES=3 python evaluate_probabilities.py --model /net/data_ttk/koller/model_data/model_data_ZToNuNu_10M_train_ev_30_ep/model_best.pt  --data /net/data_ttk/koller/model_data/model_data_{name}_10M_train_ev_30_ep/samples_test_10M_k5000.h5 --tag {name}_{n}_const_{n_name}_ev --num_const {n}  --num_events 200000 --fixed_samples')
            os.system(f'CUDA_VISIBLE_DEVICES=3 python evaluate_probabilities.py --model /net/data_ttk/koller/model_data/model_data_{name}_10M_train_ev_30_ep/model_best.pt  --data /net/data_ttk/koller/model_data/model_data_ZToNuNu_10M_train_ev_30_ep/samples_test_10M_k5000.h5 --tag ZToNuNu_{n}_const_{n_name}_ev --num_const {n}  --num_events 200000 --fixed_samples')
        if n==128:
            data_data_samples=np.load(f'/net/data_ttk/koller/model_data/model_data_{name}_10M_train_ev_30_ep/results_{name_s}_samples.npz')['probs']#[:200000]
            print(np.shape(data_data_samples))
            qcd_data_samples=np.load(f'/net/data_ttk/koller/model_data/model_data_ZToNuNu_10M_train_ev_30_ep/results_{name_s}_samples.npz')['probs']#[:200000]
            data_qcd_samples=np.load(f'/net/data_ttk/koller/model_data/model_data_{name}_10M_train_ev_30_ep/results_qcd_samples.npz')['probs']#[:200000]
            qcd_qcd_samples=np.load(f'/net/data_ttk/koller/model_data/model_data_ZToNuNu_10M_train_ev_30_ep/results_qcd_samples.npz')['probs']#[:200000]
        else:
            data_data_samples=np.load(f'/net/data_ttk/koller/model_data/model_data_{name}_10M_train_ev_30_ep/results_{name}_{n}_const_{n_name}_ev.npz')['probs']
            qcd_data_samples=np.load(f'/net/data_ttk/koller/model_data/model_data_ZToNuNu_10M_train_ev_30_ep/results_{name}_{n}_const_{n_name}_ev.npz')['probs']
            data_qcd_samples=np.load(f'/net/data_ttk/koller/model_data/model_data_{name}_10M_train_ev_30_ep/results_ZToNuNu_{n}_const_{n_name}_ev.npz')['probs']
            qcd_qcd_samples=np.load(f'/net/data_ttk/koller/model_data/model_data_ZToNuNu_10M_train_ev_30_ep/results_ZToNuNu_{n}_const_{n_name}_ev.npz')['probs']
        print(len(data_data_samples),len(data_qcd_samples))
        s_data_list[k]=[data_data_samples[o]-qcd_data_samples[o] for o in range(len(data_data_samples))]
        s_qcd_list[k]=[data_qcd_samples[p]-qcd_qcd_samples[p] for p in range(len(data_qcd_samples))]
    pl.roc_curve_comp_const(s_data_list,s_qcd_list,n_const,name)