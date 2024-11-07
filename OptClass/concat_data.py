
import pandas as pdf



maindir=''
data_file_1=maindir+''
tmp_1 = pd.read_hdf(data_file_1, key="discretized", stop=None)

data_file_2=maindir+''
tmp_2 = pd.read_hdf(data_file_2, key="discretized", stop=None)


tmp_all=pd.concat([tmp_1,tmp_2],axis=0)

print(tmp_all.shape)


tmp_all.to_hdf(maindir+'samples_train_10M.h5', key='discretized')

tmp_all = pd.read_hdf(maindir+'samples_train_10M.h5', key="discretized", stop=None)

tmp_all =tmp_all.to_numpy()[:, :600].reshape(len(tmp_all), -1, 3)
print(tmp_all)
print(np.shape(tmp_all))
