#A wrapper script to activate your python environment before running your code.
#!/bin/bash

#Activate the Python environment
source /home/home3/institut_thp/hreyes/anaconda3/envs/torch_env_pip/bin/activate 

#Run the Python script
python train_classifier_bl_scan_onebinning.py
