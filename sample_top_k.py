import os
from tqdm import tqdm

classes=['TTBar','ZToQQ','WToQQ'] #,] #,'HToGG']'ZToNuNu'
k_val=['1000','2000','2500','3000','4000','5500','6000']
n_k=len(k_val)
for c in classes:
    print(c)
    if c=='WToQQ':
        k_val=['1000','2000','2500','3000','5500','6000']
    for k in tqdm(range(n_k), total=n_k, desc=""):
        os.system('CUDA_VISIBLE_DEVICES=0 python sample_jets.py --model_dir /net/data_ttk/koller/model_data/model_data_'+c+'_10M_train_ev_30_ep --model_name model_best.pt --savetag test_top_'+k_val[k]+' --num_samples 200000 --num_const 128 --trunc '+k_val[k]+' --batchsize 100')
