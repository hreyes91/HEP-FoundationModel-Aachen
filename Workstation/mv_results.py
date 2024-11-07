import os



dirs_path=''
results=os.listdirs(dirs_path)
os.makedirs(dirs_path+'/scan_1/',exist_ok=True)
for result in results:

    sufix=result.split('_')[-1]
    if len(sufix)==7:
        os.system('mv '+dirs_path+' '+dirs_path+'/scan_1/')


