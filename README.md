
# HEP-FoundationModel-Aachen


Foundation models for High Energy Physics.


*Installation*

conda env create --name new_env_name --file torch_env_pip_env.yml


*Train backbone*


python train.py --data_path /net/data_ttk/hreyes/JetClass/discretized/TTBar_train___1Mfromeach_403030.h5 --model_path  <output-path> --log_dir <output-path>  --output linear --num_const 128 --num_epochs 5  --lr 0.001 --lr_decay 1e-06 --batch_size 100 --num_events 1000 --dropout 0 --num_heads 4 --num_layers 8 --num_bins 41 31 31 --weight_decay 1e-05 --hidden_dim 256 --end_token --start_token  --name_sufix YONFFAQ --num_events_val 5000 --checkpoint_steps 1200000


