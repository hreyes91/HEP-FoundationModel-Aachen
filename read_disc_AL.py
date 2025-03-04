

import h5py
import numpy as np
import matplotlib.pyplot as plt

def read_h5_file(file_path):
    """Reads an HDF5 file and prints its contents."""
    f=h5py.File(file_path, 'r')
    print("Keys: %s" % f.keys())
    # get first object name/key; may or may NOT be a group
    #a_group_key = list(f.keys())[0]
        
    #print(f.get('jet1').keys())

    #print(f.get('jet_coords'))
    #print(np.array(f.get('jet1').get('4mom')))

    return f

file_path='LHCO_discrete/discrete_Weak-mix-Train-100_1Mfromeach_403030.h5'


f=read_h5_file(file_path)

jet1=f.get('discretized_jet1').get('axis1')[:]
print(jet1)
