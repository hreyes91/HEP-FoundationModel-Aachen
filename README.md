<<<<<<< HEAD
# HEP-FoundationModel-Aachen
=======
Install Dependencies for GPU (with anaconda):
```
conda create --name torch_env python=3.9
conda activate torch_env
conda install pytorch cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
```
Change "+cu113" to "+cpu" if you have no gpu


To train a model run:
```
python train.py
```

Validation is coming soon...
>>>>>>> humberto
