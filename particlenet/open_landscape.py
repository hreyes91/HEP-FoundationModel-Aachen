
import pandas as pd
import h5py

# Open the HDF5 file in read mode
file_path = '/home/home3/institut_thp/hreyes/Transformers/transformers_finke/datasets/TopQuarkTag/topquarktagging_data/discretized//train_qcd_40_30_30_forall.h5'



dat_train = pd.read_hdf(file_path, key='raw')
dat_train_disc = pd.read_hdf(file_path, key='discretized')
print(dat_train)
print(dat_train_disc)

file_path_val = '/home/home3/institut_thp/hreyes/Transformers/transformers_finke/datasets/TopQuarkTag/topquarktagging_data/discretized//val_qcd_40_30_30_forall.h5'
dat_val = pd.read_hdf(file_path_val, key='raw')
dat_val_disc = pd.read_hdf(file_path_val, key='discretized')
print(dat_val)


file_path_all = '/home/home3/institut_thp/hreyes/Transformers/transformers_finke/datasets/TopQuarkTag/topquarktagging_data/discretized//all_qcd_40_30_30_forall.h5'
data_all=pd.concat([dat_train,dat_val],axis=0)
print(data_all.shape)

data_all_disc=pd.concat([dat_train_disc,dat_val_disc],axis=0)

#data_all.to_csv(file_path_all,index=False)

data_all.to_hdf(file_path_all, key="raw", mode="w", complevel=9)
data_all_disc.to_hdf(file_path_all, key="discretized", mode="r+", complevel=9)


print(data_all)
