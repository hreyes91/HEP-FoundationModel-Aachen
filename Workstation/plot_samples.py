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

def Make_Plots(jets,pt_bins,eta_bins,phi_bins,mj,jets_true,ptj_true,mj_true,path_to_plots):

    plt.hist(np.log(jets[:,:,0]).flatten(), bins=pt_bins, color='blue',histtype='step',density=True)
    plt.hist(np.log(jets_true[:,:,0]).flatten(), bins=pt_bins, color='red',histtype='step',density=True)
    plt.xlabel('$\log (p_T)$')
    
    plt.savefig(path_to_plots+'/plot_pt_trans.png')
    plt.close()
    
    plt.hist(jets[:,:,1].flatten(), bins=eta_bins, color='blue',histtype='step',density=True)
    plt.hist(jets_true[:,:,1].flatten(), bins=eta_bins, color='red',histtype='step',density=True)
    plt.xlabel('$\Delta\eta$')
    plt.savefig(path_to_plots+'/plot_eta_trans.png')
    plt.close()
    
    plt.hist(jets[:,:,2].flatten(), bins=phi_bins, color='blue',histtype='step',density=True)
    plt.hist(jets_true[:,:,2].flatten(), bins=phi_bins, color='red',histtype='step',density=True)
    plt.xlabel('$\Delta\phi$')
    plt.savefig(path_to_plots+'/plot_phi_trans.png')
    plt.close()
    
    mask = jets[:, :, 0] != 0
    plt.hist(np.sum(mask, axis=1), bins=np.linspace(-0.5, 100.5, 102),color='blue',histtype='step',density=True)
    mask = jets_true[:, :, 0] != 0
    plt.hist(np.sum(mask, axis=1), bins=np.linspace(-0.5, 100.5, 102),color='red',histtype='step',density=True)
    plt.xlabel('Multiplicity')
    plt.savefig(path_to_plots+'/plot_mul_trans.png')
    plt.close()
    
    
    mj_bins = np.linspace(0, 1, 100)
    plt.hist(np.clip(mj, mj_bins[0], mj_bins[-1]), bins=mj_bins,color='blue',histtype='step',density=True)
    plt.hist(np.clip(mj_true, mj_bins[0], mj_bins[-1]), bins=mj_bins,color='red',histtype='step',density=True)
    plt.xlabel('$m_{jet}$')
    plt.savefig(path_to_plots+'/plot_mj_trans.png')
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




#test_results_dir='/Users/humbertosmac/Documents/work/Transformers/Transformers_finke/test_results/'
main_dir='/net/data_t2k/transformers-hep/JetClass/'
test_results_dir=main_dir+'/TTBar_models/'
test_data_name='/discretized/TTBar_test___10M_TTBar.h5'

bins_path_prefix='../preprocessing_bins/'
bin_tag='10M_TTBar'

tag_oftrain='TTBar_run_testwall_10M'
tag_foreval='test_eval_200k'
tag_foreval_other='test_eval_other_200k'
tag_forsample='test_sample_200k_trunc'

pt_bins = np.load(bins_path_prefix+'pt_bins_'+bin_tag+'.npy')
eta_bins = np.load(bins_path_prefix+'eta_bins_'+bin_tag+'.npy')
phi_bins = np.load(bins_path_prefix+'phi_bins_'+bin_tag+'.npy')

n_samples=200000
discrete_truedata_filename=main_dir+test_data_name
jets_true,ptj_true,mj_true=LoadTrue(discrete_truedata_filename,n_samples)
print(mj_true)


n_trains=6
for j in range(4,n_trains):


    
    filename=test_results_dir+'/'+tag_oftrain+'_'+str(j)+'/samples_'+tag_forsample+'.h5'

    path_to_plots=test_results_dir+'/'+tag_oftrain+'_'+str(j)+'/'

    #filename = test_results_dir+"/ttbar_run_b_2_6/samples_test_sample_200k.h5"
    tmp = pd.read_hdf(filename, key="discretized", stop=None)
    
    
    tmp = tmp.to_numpy()[:, :600].reshape(len(tmp), -1, 3)
    print(tmp.shape)

    mask = tmp[:, :, 0] == -1
    print(mask)
    jets,ptj,mj = make_continues(tmp, mask,pt_bins,eta_bins,phi_bins, noise=False)

    Make_Plots(jets,pt_bins,eta_bins,phi_bins,mj,jets_true,ptj_true,mj_true,path_to_plots)





# /net/data_t2k/transformers-hep/JetClass/TTBar_models/TTBar_run_testwall_10M_4/samples_test_sample_200k_trunc.h5
