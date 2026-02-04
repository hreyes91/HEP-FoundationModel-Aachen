#!/bin/bash
#Activate the Python environment
#source /home/home3/institut_thp/hreyes/anaconda3/envs/torch_env_pip/bin/activate 

source /home/home3/institut_thp/hreyes/anaconda3/etc/profile.d/conda.sh

# Activate your environment
conda activate torch_env_pip

#Run the Python script

cd /home/home3/institut_thp/hreyes/Transformers/FoundationModels/HEP-FoundationModel-Aachen

run="RUN101"


python train.py --data_path /net/data_ttk/hreyes/OneBin/ZJetsToNuNu_train___1Mfromeach_403030.h5 --model_path  /net/data_ttk/hreyes/FLJT/ResultsV2/Tests_QCD --log_dir /net/data_ttk/hreyes/FLJT/ResultsV2/Tests_QCD  --output linear --num_const 128 --num_epochs 36  --lr 0.001 --lr_decay 1e-06 --batch_size 100 --num_events 10000000 --dropout 0 --num_heads 4 --num_layers 8 --num_bins 41 31 31 --weight_decay 1e-05 --hidden_dim 256 --end_token --start_token  --name_sufix ${run} --num_events_val 500000 --checkpoint_steps 1200000


myArray=(0 1 5 10 15 20 25 30 35)

for str in "${myArray[@]}"; do
    echo "Running training with $str events..."

 

    echo "Sampling jets for $str..."
    python sample_jets.py --model_dir  /net/data_ttk/hreyes/FLJT/ResultsV2/Tests_QCD_${run}/ --savetag epoch_${str}  --num_samples 100 --num_const 128 --trunc 5000 --batchsize 100 --model_name model_epoch_${str}.pt
    echo "density estimation jets for $str..."
    python evaluate_probabilities.py --model  /net/data_ttk/hreyes/FLJT/ResultsV2/Tests_QCD_${run}/model_epoch_${str}.pt  --data /net/data_ttk/hreyes/FLJT/ResultsV2/Tests_QCD_${run}/samples_epoch_${str}.h5 --tag eval_samples_epoch_${str} --num_const 128  --num_events 100 --fixed_samples

done
