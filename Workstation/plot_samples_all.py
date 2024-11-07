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


    plt.hist(jets_gen[:,:,1].flatten(), bins=eta_bins, color='black',histtype='step',density=True,label="Gen")
    plt.hist(jets_true[:,:,1].flatten(), bins=eta_bins, color='red',histtype='step',density=True,label="True")
    plt.xlabel('$\Delta\eta$')
    plt.legend()
    plt.title(plot_title,loc='left')
    plt.savefig(path_to_plots+'plot_eta_trans_all.png')
    plt.close()
    
    plt.hist(jets_gen[:,:,2].flatten(), bins=phi_bins, color='black',histtype='step',density=True,label="Gen")
    plt.hist(jets_true[:,:,2].flatten(), bins=phi_bins, color='red',histtype='step',density=True,label="True")
    plt.xlabel('$\Delta\phi$')
    plt.legend()
    plt.title(plot_title,loc='left')
    plt.savefig(path_to_plots+'plot_phi_trans_all.png')
    
    plt.close()
    
    mask = jets_gen[:, :, 0] != 0
    plt.hist(np.sum(mask, axis=1), bins=np.linspace(-0.5, 100.5, 102),color='blue',histtype='step',density=True,label="Gen")
    
    

    
    
    mask = jets_true[:, :, 0] != 0
    plt.hist(np.sum(mask, axis=1), bins=np.linspace(-0.5, 100.5, 102),color='red',histtype='step',density=True,label="True")
    plt.legend()
    plt.title(plot_title,loc='left')
    plt.xlabel('Multiplicity')
    plt.savefig(path_to_plots+'plot_mul_trans_all.png')
    plt.close()
    
    
    mj_bins = np.linspace(0, 450, 100)
    

    plt.hist(np.clip(mj_gen, mj_bins[0], mj_bins[-1]), bins=mj_bins,color='black',histtype='step',density=True,label="Gen")
    
    
    plt.hist(np.clip(mj_true, mj_bins[0], mj_bins[-1]), bins=mj_bins,color='red',histtype='step',density=True,label="True")
    
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

list_of_jets=['TTBar','HToBB','HToGG','HToWW2Q1L','HToWW4Q','TTBarLep','ZToQQ']


dir_of_test_data='/net/data_t2k/transformers-hep/JetClass/discretized/'
main_dir='/net/data_t2k/transformers-hep/JetClass/All_models_for_OptClass/'
bins_path_prefix='../preprocessing_bins/'
n_samples=200000
for jet_name in list_of_jets:


    
    pt_bins = np.load(bins_path_prefix+'pt_bins_10M_'+jet_name+'.npy')
    eta_bins = np.load(bins_path_prefix+'eta_bins_10M_'+jet_name+'.npy')
    phi_bins = np.load(bins_path_prefix+'phi_bins_10M_'+jet_name+'.npy')

    test_data_dir=main_dir+'/all_const_403030_'+jet_name+'/'
    test_data_gen =test_data_dir+'/samples_samples_nsamples200000_trunc_5000.h5'
    
    path_to_plots=test_data_dir
    
    discrete_truedata_filename=dir_of_test_data+jet_name+'_test___10M_'+jet_name+'.h5'
    #plot_title="$g/q $"
    plot_title=jet_name
    jets_gen,ptj_gen,mj_gen=LLM(test_data_gen)


    jets_true,ptj_true,mj_true=LoadTrue(discrete_truedata_filename,n_samples)
    print(mj_true)

    Make_Plots(jets_gen,mj_gen,pt_bins,eta_bins,phi_bins,jets_true,ptj_true,mj_true,path_to_plots,plot_title)



