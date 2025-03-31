

import os


#list_of_files=['Weak-mix-Train-100.h5','Weak-mix-Train-1000.h5','Weak-mix-Train-10000.h5','Weak-mix-Train-2000.h5','Weak-mix-Train-300.h5','Weak-mix-Train-5000.h5','Weak-mix-Train-600.h5','bg-N100-SR-Train.h5','bg-N100-SR-Train-dat.h5','bg-N100-SR-Test.h5','sn-N100-SR-Test.h5']

list_of_files=['bg-N100-SR-Test_150k.h5']


#list_of_files=['Weak-mix-Train-100.h5']
for file in list_of_files:
    print(file)
    
    file_name=file.split('.')[0]
    command='python preprocess_LHCO_onebinner.py --input_file /Users/humbertosmac/Documents/work/Foundation_Model/AnomalyDetection/LHCOData/processed_jg/original/ReasembleAL_1/'+str(file)+' --nBins 40 30 30 --nJets 500000 --tag 1Mfromeach_403030'

    os.system(command)


