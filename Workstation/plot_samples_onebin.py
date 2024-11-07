import os
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt




def make_continues(jets, mask,pt_bins,eta_bins,phi_bins, noise=False):


    pt_disc = jets[:, :, 0]
    eta_disc = jets[:, :, 1]
    phi_disc = jets[:, :, 2]

    if noise:
        pt_con = (pt_disc - np.random.uniform(0.0, 1.0, size=pt_disc.shape)) * (
            pt_bins[1] - pt_bins[0]
        ) + pt_bins[0]
        eta_con = (eta_disc - np.random.uniform(0.0, 1.0, size=eta_disc.shape)) * (
            eta_bins[1] - eta_bins[0]
        ) + eta_bins[0]
        phi_con = (phi_disc - np.random.uniform(0.0, 1.0, size=phi_disc.shape)) * (
            phi_bins[1] - phi_bins[0]
        ) + phi_bins[0]
    else:
        pt_con = (pt_disc - 0.5) * (pt_bins[1] - pt_bins[0]) + pt_bins[0]
        eta_con = (eta_disc - 0.5) * (eta_bins[1] - eta_bins[0]) + eta_bins[0]
        phi_con = (phi_disc - 0.5) * (phi_bins[1] - phi_bins[0]) + phi_bins[0]


    pt_con = np.exp(pt_con)
    pt_con[mask] = 0.0
    eta_con[mask] = 0.0
    phi_con[mask] = 0.0
    
    pxs = np.cos(phi_con) * pt_con
    pys = np.sin(phi_con) * pt_con
    pzs = np.sinh(eta_con) * pt_con
    es = (pxs ** 2 + pys ** 2 + pzs ** 2) ** (1. / 2)

    pxj = np.sum(pxs, -1)
    pyj = np.sum(pys, -1)
    pzj = np.sum(pzs, -1)
    ej = np.sum(es, -1)
    
    ptj = np.sqrt(pxj**2 + pyj**2)
    mj = (ej ** 2 - pxj ** 2 - pyj ** 2 - pzj ** 2) ** (1. / 2)

    continues_jets = np.stack((pt_con, eta_con, phi_con), -1)

    return continues_jets, ptj, mj

def Make_Plots(jets_gen,mj_gen,pt_bins,eta_bins,phi_bins,jets_true,ptj_true,mj_true,path_to_plots,plot_title):
    print('making plots')
    plt.hist(np.log(jets_gen[:,:,0]).flatten(), bins=pt_bins, color='black',histtype='step',density=True,label="Gen")
    plt.hist(np.log(jets_true[:,:,0]).flatten(), bins=pt_bins, color='red',histtype='step',density=True,label="True")
    plt.legend()
    plt.title(plot_title,loc='left')
    plt.xlabel('$\log (p_T)$')
    
    plt.savefig(path_to_plots+'plot_pt_trans_all.png')
    plt.close()

    mask_tr = jets_true[:, :, 0] != 0
    mask_eta_true = jets_true[mask_tr, 1]
    
    mask_gen = jets_gen[:, :, 0] != 0
    mask_eta_gen = jets_gen[mask_gen, 1]
    
    
    plt.hist(mask_eta_gen, bins=eta_bins, color='black',histtype='step',density=True,label="Gen")
    plt.hist(mask_eta_true, bins=eta_bins, color='red',histtype='step',density=True,label="True")
    plt.xlabel('$\Delta\eta$')
    plt.legend()
    plt.title(plot_title,loc='left')
    plt.savefig(path_to_plots+'plot_eta_trans_all.png')
    plt.close()
    
    
    mask_tr = jets_true[:, :, 0] != 0
    mask_phi_true = jets_true[mask_tr,2]
    
    mask_gen = jets_gen[:, :, 0] != 0
    mask_phi_gen = jets_gen[mask_gen,2]
    
    plt.hist(mask_phi_gen, bins=phi_bins, color='black',histtype='step',density=True,label="Gen")
    plt.hist(mask_phi_true, bins=phi_bins, color='red',histtype='step',density=True,label="True")
    plt.xlabel('$\Delta\phi$')
    plt.legend()
    plt.title(plot_title,loc='left')
    plt.savefig(path_to_plots+'plot_phi_trans_all.png')
    
    plt.close()
    
    mask = jets_gen[:, :, 0] != 0
    plt.hist(np.sum(mask, axis=1), bins=np.linspace(-0.5, 100.5, 102),color='blue',histtype='step',density=True,label="Gen")
    
    

    
    
    mask = jets_true[:, :, 0] != 0
    plt.hist(np.sum(mask, axis=1), bins=np.linspace(-0.5, 150.5, 152),color='red',histtype='step',density=True,label="True")
    plt.legend()
    plt.title(plot_title,loc='left')
    plt.xlabel('Multiplicity')
    plt.savefig(path_to_plots+'plot_mul_trans_all.png')
    plt.close()
    
    
    mj_bins = np.linspace(0, 450, 100)
    

    plt.hist(np.clip(mj_gen, mj_bins[0], mj_bins[-1]), bins=mj_bins,color='black',histtype='step',density=True,label="Gen")
    
    
    plt.hist(np.clip(mj_true, mj_bins[0], mj_bins[-1]), bins=mj_bins,color='red',histtype='step',density=True,label="True")
    plt.yscale('log')
    
    plt.legend()
    plt.title(plot_title,loc='left')
    plt.xlabel('$m_{jet}$')
    plt.savefig(path_to_plots+'plot_mj_trans_all.png')
    plt.close()
    return


def LoadTrue(discrete_truedata_filename,n_samples):


    tmp = pd.read_hdf(discrete_truedata_filename, key="discretized", stop=None)
    print(tmp)
    print(tmp.shape)
    
    tmp = tmp.to_numpy()[:, :600].reshape(len(tmp), -1, 3)
    print(tmp)
    print(tmp.shape)
    print('hello')
    tmp=tmp[:n_samples,:,:]
    print(tmp.shape)
    print('hello')
 

    mask = tmp[:, :, 0] == -1
    print(mask)
    jets_true,ptj_true,mj_true = make_continues(tmp, mask,pt_bins,eta_bins,phi_bins, noise=False)

    return jets_true,ptj_true,mj_true

def LLM(filename):

    tmp = pd.read_hdf(filename, key="discretized", stop=None)
    
    
    tmp = tmp.to_numpy()[:, :600].reshape(len(tmp), -1, 3)
    print(tmp.shape)

    mask = tmp[:, :, 0] == -1
    print(mask)
    jets,ptj,mj = make_continues(tmp, mask,pt_bins,eta_bins,phi_bins, noise=False)

    return jets,ptj,mj

#list_of_jets=['TTBar','ZJetsToNuNu','HToBB','HToCC','HToGG','HToWW2Q1L','HToWW4Q','TTBarLep','WToQQ','ZToQQ']


list_of_jets=['TTBar','ZJetsToNuNu']

dict_of_names={'TTBar':r"$t \rightarrow bqq^{\prime} $",'ZJetsToNuNu':r"$g/q$",'HToBB':r"$H\rightarrow b\bar b $",'HToCC':r"$H \rightarrow c \bar c$",'HToGG':r"$H \rightarrow gg$",'HToWW2Q1L':r"$H \rightarrow lvqq^{\prime}$",'HToWW4Q':r"$H \rightarrow 4q$",'TTBarLep':r"$H \rightarrow blv$",'WToQQ':r"$W \rightarrow  qq^{\prime}$",'ZToQQ':r"$Z \rightarrow q \bar q$"}

dir_of_test_data='/net/data_t2k/transformers-hep/JetClass/discretized/'
main_dir='/net/data_t2k/transformers-hep/JetClass/OptClass/'
bins_path_prefix='../preprocessing_bins/'
n_samples=200000
trunc_list=[5000]
bin_tag='1Mfromeach_403030'
for jet_name in list_of_jets:


    
    pt_bins = np.load(bins_path_prefix+'pt_bins_'+bin_tag+'.npy')
    eta_bins = np.load(bins_path_prefix+'eta_bins_'+bin_tag+'.npy')
    phi_bins = np.load(bins_path_prefix+'phi_bins_'+bin_tag+'.npy')
    
    test_data_dir=main_dir+jet_name+'_models/'
    
    gen_dirs=os.listdir(test_data_dir)
    
    for gen_dir in gen_dirs:
    
        if '1Mfromeach_403030' not in gen_dir:
            continue
    
    
    
        discrete_truedata_filename=dir_of_test_data+jet_name+'_test___1Mfromeach_403030.h5'
        #plot_title="$g/q $"
        plot_title=dict_of_names.get(jet_name)

        jets_true,ptj_true,mj_true=LoadTrue(discrete_truedata_filename,n_samples)
        print(mj_true)
    
    
        for trunc in trunc_list:
        
            path_to_plots=test_data_dir+gen_dir+'/'+'log_mj_trunc'+str(trunc)+str(n_samples)+'/'
            os.makedirs(path_to_plots,exist_ok=True)
            test_data_gen =test_data_dir+gen_dir+'/samples__nsamples'+str(n_samples)+'_trunc_'+str(trunc)+'.h5'
            
            jets_gen,ptj_gen,mj_gen=LLM(test_data_gen)
            Make_Plots(jets_gen,mj_gen,pt_bins,eta_bins,phi_bins,jets_true,ptj_true,mj_true,path_to_plots,plot_title)
        


