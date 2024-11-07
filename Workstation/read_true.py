import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt




def ReadTopTagData(filename):

    toptagdata = pd.read_hdf(filename, key="table", stop=None)

    return toptagdata



def ReadJetClass(filename):

    topjetclass = pd.read_hdf(filename, key="raw", stop=None)

    return  topjetclass



#filename_toptagg='/Users/humbertosmac/Documents/work/Transformers/Transformers_finke/datasets/toptag_data/val.h5'
#toptagdata=ReadTopTagData(filename_toptagg)
#print(toptagdata.head(10))

filename_jetclass='/net/data_t2k/transformers-hep/JetClass/test/TTBar_test.h5'
topjetclass=ReadJetClass(filename_jetclass)
print(topjetclass.shape)
print( topjetclass.head(10))
print( topjetclass.tail(10))
exit()
numpy_topjetclass = topjetclass.to_numpy()

print(numpy_topjetclass)
print(np.shape(numpy_topjetclass))


pt_topjetclass=numpy_topjetclass[:,::3].copy()
print(pt_topjetclass)
print(np.shape(pt_topjetclass))

eta_topjetclass=numpy_topjetclass[:,1::3]
print(eta_topjetclass)
print(np.shape(eta_topjetclass))

phi_topjetclass=numpy_topjetclass[:,2::3]
print(phi_topjetclass)
print(np.shape(phi_topjetclass))
