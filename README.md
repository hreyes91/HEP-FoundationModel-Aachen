
# HEP-FoundationModel-Aachen


Foundation models for High Energy Physics.



 # Data preprocessing
 
 To be filled. For now use already preprocessed data in /net/data_ttk/hreyes/JetClass/discretized/OneBin

 # Installation

```bash
conda env create --name new_env_name --file torch_env_pip_env.yml

```
# Train backbone

```python

python train.py --data_path /net/data_ttk/hreyes/JetClass/discretized/OneBin/TTBar_train___1Mfromeach_403030.h5 --model_path  <model-dir> --log_dir <model-dir>  --output linear --num_const 128 --num_epochs 5  --lr 0.001 --lr_decay 1e-06 --batch_size 100 --num_events 1000 --dropout 0 --num_heads 4 --num_layers 8 --num_bins 41 31 31 --weight_decay 1e-05 --hidden_dim 256 --end_token --start_token  --name_sufix YONFFAQ --num_events_val 5000 --checkpoint_steps 1200000

```

# sample jets

```python
python sample_jets.py --model_dir <model-dir> --savetag <samples-tag-name>  --num_samples 100 --num_const 128 --trunc 5000 --batchsize 100 --model_name best
```

# density estimation

```python

python evaluate_probabilities.py --model <model-dir>+/model_best.pt +' --data /net/data_ttk/hreyes/JetClass/discretized/OneBin/TTBar_test___1Mfromeach_403030.h5 --tag <evals-tag-name> --num_const 128  --num_events 128 --fixed_samples

```
