import h5py
import pandas as pd

file_path='/net/data_t2k/transformers-hep/JetClass/discretized/TTBar_test___40_30_30_pt_part_TTBar.h5'
data_h5= h5py.File(file_path, 'r')


print(list(data_h5.keys()))


frame=pd.read_hdf(file_path, key='discretized', stop=5)
print(frame)
