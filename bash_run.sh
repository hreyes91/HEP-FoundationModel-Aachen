#A wrapper script to activate your python environment before running your code.
#!/bin/bash

#Activate the Python environment
#source /home/home3/institut_thp/hreyes/anaconda3/envs/torch_env_pip/bin/activate 
source /home/home3/institut_thp/hreyes/anaconda3/etc/profile.d/conda.sh

# Activate your environment
conda activate torch_env_pip



#Run the Python script
python train_classifier_scan_Al_finetune.py
