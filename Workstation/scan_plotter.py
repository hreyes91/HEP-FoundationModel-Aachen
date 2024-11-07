


####Plot samples sector

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

def Make_Plots(jets,pt_bins,eta_bins,phi_bins,mj,jets_true,ptj_true,mj_true,path,path_to_results):

    plt.hist(np.log(jets[:,:,0]).flatten(), bins=pt_bins, color='blue',histtype='step',density=True)
    plt.hist(np.log(jets_true[:,:,0]).flatten(), bins=pt_bins, color='red',histtype='step',density=True)
    plt.xlabel('$\log (p_T)$')
    
    plt.savefig(path_to_results+'/plot_pt_trans.png')
    plt.close()
    
    plt.hist(jets[:,:,1].flatten(), bins=eta_bins, color='blue',histtype='step',density=True)
    plt.hist(jets_true[:,:,1].flatten(), bins=eta_bins, color='red',histtype='step',density=True)
    plt.xlabel('$\Delta\eta$')
    plt.savefig(path_to_results+'/plot_eta_trans.png')
    plt.close()
    
    plt.hist(jets[:,:,2].flatten(), bins=phi_bins, color='blue',histtype='step',density=True)
    plt.hist(jets_true[:,:,2].flatten(), bins=phi_bins, color='red',histtype='step',density=True)
    plt.xlabel('$\Delta\phi$')
    plt.savefig(path_to_results+'/plot_phi_trans.png')
    plt.close()
    
    mask = jets[:, :, 0] != 0
    plt.hist(np.sum(mask, axis=1), bins=np.linspace(-0.5, 100.5, 102),color='blue',histtype='step',density=True)
    mask = jets_true[:, :, 0] != 0
    plt.hist(np.sum(mask, axis=1), bins=np.linspace(-0.5, 100.5, 102),color='red',histtype='step',density=True)
    plt.xlabel('Multiplicity')
    plt.savefig(path_to_results+'/plot_mul_trans.png')
    plt.close()
    
    
    mj_bins = np.linspace(0, 450, 100)
    plt.hist(np.clip(mj, mj_bins[0], mj_bins[-1]), bins=mj_bins,color='blue',histtype='step',density=True)
    plt.hist(np.clip(mj_true, mj_bins[0], mj_bins[-1]), bins=mj_bins,color='red',histtype='step',density=True)
    plt.xlabel('$m_{jet}$')
    plt.savefig(path_to_results+'/plot_mj_trans.png')
    plt.close()
    return


def LoadTrue(discrete_truedata_filename):


    tmp = pd.read_hdf(discrete_truedata_filename, key="discretized", stop=None)
    print(tmp)
    print(tmp.shape)
    tmp = tmp.to_numpy()[:, :300].reshape(len(tmp), -1, 3)
    print(tmp)
    print(tmp.shape)
    
    exit()
    mask = tmp[:, :, 0] == -1
    print(mask)
    jets_true,ptj_true,mj_true = make_continues(tmp, mask,pt_bins,eta_bins,phi_bins, noise=False)

    return jets_true,ptj_true,mj_true

def Params_for_sampleplots(LLM_filename,discrete_truedata_filename=):
    LLM_filename = "test_results/full_test_qcd/_1/samples_test_qcd_1_200000samplestopk5k.h5"
    discrete_truedata_filename='../../datasets/JetClass/discretized/train_TTBar__top_JetClassttbar_1.h5'
    return

def LoadLLMdata(filename):

    tmp = pd.read_hdf(filename, key="discretized", stop=None)
    
    
    tmp = tmp.to_numpy()[:, :300].reshape(len(tmp), -1, 3)
    print(tmp.shape)
    mask = tmp[:, :, 0] == -1
    print(mask)
    jets,ptj,mj = make_continues(tmp, mask,pt_bins,eta_bins,phi_bins, noise=False)


    return jets,ptj,mj

def LoadBins(bins_path_prefix,bins_tag):
    bins_path_prefix='test_results/preprocessing_bins/'
    pt_bins = np.load(bins_path_prefix+'pt_bins_None.npy')
    eta_bins = np.load(bins_path_prefix+'eta_bins_None.npy')
    phi_bins = np.load(bins_path_prefix+'phi_bins_None.npy')

    return pt_bins,eta_bins,phi_bins

def DoSamplePlots(discrete_truedata_filename,LLM_filename,path_to_results,pt_bins,eta_bins,phi_bins):

    
    jets,ptj,mj=LoadLLMdata(LLM_filename)
    jets_true,ptj_true,mj_true=LoadTrue(discrete_truedata_filename)
    Make_Plots(pt_bins,eta_bins,phi_bins,jets,ptj,mj,jets_true,ptj_true,mj_true,path_to_results)


    return


####probability evaluation section

def GetDataFromQCDT(file_dir, evalprob_topfromqcd,eval_tag_qcdfromqcd):

    file_name_qcd='evalprob_topfromqcd'
    file_name_top='revalprob_qcdfromqcd'
    

    file_qcd=file_dir+'/'+file_name_qcd
    evalprob_qcdfromqcd = np.load(file_qcd)
    
    file_top=file_dir+'/'+file_name_top
    evalprob_topfromqcd = np.load(file_top)



    return evalprob_qcdfromqcd,evalprob_topfromqcd


def GetDataFromTopT(file_dir, evalprob_topfromtop,eval_tag_qcdfromtop):

    
    file_name_top=evalprob_topfromtop
    file_name_qcd=evalprob_qcdfromtop
    

    file_top=file_dir+'/'+file_name_top
    evalprob_topfromtop = np.load(file_top)
    
    file_qcd=file_dir+'/'+file_name_qcd
    evalprob_qcdfromtop = np.load(file_qcd)
    
    return evalprob_topfromtop,evalprob_qcdfromtop


def ExpectedProb(evalprob):
    exp_logp=np.log(1/(39402))*evalprob['n_const']
    return exp_logp

def ComputeLLR(evalprobT,evalprobF,type):

    s=evalprobT['probs']-evalprobF['probs']

    return s

def PlotLLR(s_qcd,s_top):
    bins = np.linspace(-100, 120, 40)
    plt.hist(s_qcd,histtype='step',bins=bins,density=True,color='blue',label='QCD')
    plt.hist(s_top,histtype='step',bins=bins,density=True,color='black',label='Top')
    plt.legend()
    plt.savefig('plot_LLR_test_1.png')
    plt.close()
    return


def ROCcurve(s_qcd,s_top):
    from sklearn.metrics import roc_curve, roc_auc_score
    fpr, tpr, _ = roc_curve(np.append(np.zeros(len(s_qcd)), np.ones(len(s_top))), np.append(s_qcd, s_top))
    print(roc_auc_score(np.append(np.zeros(len(s_qcd)), np.ones(len(s_top))), np.append(s_qcd, s_top)))
    print(fpr)
    plt.plot(fpr,tpr, label="Transformer", c="r")
    plt.savefig('plot_ROC_1.png')
    plt.close()
    
    plt.plot(tpr,1/fpr, label="Transformer", c="r")
    plt.ylim(1, 1e4)
    plt.xlim(0, 1)
    plt.xlabel(r"$\epsilon_{\rm{top}}$")
    plt.ylabel(r"$1 / \epsilon_{\rm{QCD}}$")
    plt.yscale('log')
    #plt.xscale('log')
    plt.savefig('plot_ROC2_1.png')
    plt.close()
    return

def plot_probs(evalprob_qcdfromqcd,evalprob_topfromqcd,evalprob_topfromtop,evalprob_qcdfromtop):

 plt.hist(evalprob_qcdfromqcd['probs'],histtype='step',bins=30,density=True,color='blue',label='QCD')
 plt.hist(evalprob_qcdfromtop['probs'],histtype='step',bins=30,density=True,color='blue',linestyle='--')
 
 plt.hist(evalprob_topfromtop['probs'],histtype='step',bins=30,density=True,color='black',label='Top')
 plt.hist(evalprob_topfromqcd['probs'],histtype='step',bins=30,density=True,color='black',linestyle='--')
 plt.xlabel('log(p)')
 plt.legend()
 
 plt.savefig('plot_probs_test_1.png')
 plt.close()

 return



def plot_color(evalprob):

 plt.plot(evalprob['probs'],evalprob['n_const'],'.')
 plt.xlabel('n_{const}')
 plt.ylabel('log(p)')
 plt.savefig('plot_plot_test_1.png')
 plt.close()

 return
 
 
def plot_contour(evalprob_qcdfromqcd,evalprob_topfromtop):

 sns.kdeplot(data=evalprob_qcdfromqcd,x='probs',y='n_const',levels=4,color='blue', fill=True,alpha=.5)
 sns.kdeplot(data=evalprob_topfromtop,x='probs',y='n_const',levels=4,color='black', fill=True,alpha=.5)
 plt.plot(np.log(1. / 39402) * np.linspace(0, 100, 100), np.linspace(0, 100, 100), linestyle="--", color='grey')
 plt.ylabel('$n_{const}$')
 plt.xlabel('log(p)')
 plt.savefig('plot_contour_test_1.png')
 plt.close()

 return

def DoProbEvaluation(file_dir,eval_tag_topfromqcd,eval_tag_qcdfromqcd,eval_tag_topfromtop,eval_tag_qcdfromtop):


    evalprob_qcdfromqcd,evalprob_topfromqcd=GetDataFromQCDT(file_dir, eval_tag_topfromqcd,eval_tag_qcdfromqcd)
    evalprob_topfromtop,evalprob_qcdfromtop=GetDataFromTopT(file_dir, eval_tag_topfromtop,eval_tag_qcdfromtop)

    plot_probs(evalprob_qcdfromqcd,evalprob_topfromqcd,evalprob_topfromtop,evalprob_qcdfromtop)
    #plot_contour(evalprob_qcdfromqcd,evalprob_topfromtop)


    s_top=ComputeLLR(evalprob_topfromtop,evalprob_topfromqcd,'qcd')
    s_qcd=ComputeLLR(evalprob_qcdfromtop,evalprob_qcdfromqcd,'qcd')
    PlotLLR(s_qcd,s_top)
    ROCcurve(s_qcd,s_top)


    plot_color(evalprob)
    plot_contour(evalprob)

    return


def ReadArguments(dir):
    
    
    file_name=dir+'/arguments.txt'
    f = open(file_name, "r")
    lines = f.readlines()

    for line in lines:
        if 'num_events' in line:
            line=line.replace(' ','')
            num_events=int(line.split('num_events')[-1])
            print(num_events)
    return num_events

def Select_Result(scan_dirs):
    result_dir=None
    for dir in scan_dirs:
        num_events_current=ReadArguments()
        if num_events==num_events_current:
            result_dir=dir

    return result_dir



TTbar_scan_path=
qcd_scan_path=


TTbar_scan_dirs=os.listdir(TTbar_scan_path)
qcd_scan_dirs=os.listdir(qcd_scan_path)


qcd_discrete_truedata_filename=''
ttbar_discrete_truedata_filename=''


qcd_bins_path_prefix=''
ttbar_bins_path_prefix=''

qcd_pt_bins,qcd_eta_bins,qcd_phi_bins=LoadBins(qcd_bins_path_prefix,ttbar_bins_tag)
ttbar_bins,ttbar_bins,ttbar_phi_bins=LoadBins(ttbar_bins_path_prefix,ttbar_bins_tag)


####Samples section
for qcd_result_dir in qcd_scan_dirs:

    DoSamplePlots(qcd_discrete_truedata_filename,LLM_filename,path_to_results,qcd_pt_bins,qcd_eta_bins,qcd_phi_bins)


for ttbar_result_dir in qcd_scan_dirs:


    DoSamplePlots(ttbar_discrete_truedata_filename,LLM_filename,path_to_results,ttbar_pt_bins,ttbar_eta_bins,ttbar_phi_bins)


exit()
###probabilities section ###
eval_tag_qcdfromtop=
eval_tag_topfromtop=
eval_tag_qcdfromqcd=
eval_tag_topfromqcd=
Num_events_list=[]

for num_events in num_events_list:
    
    
    qcd_dir=Select_Result(qcd_scan_dirs)
    ttbar_dir=Select_Result(ttbar_scan_dirs)
    
    
    DoProbEvaluation(qcd_dir,ttbar_dir)
    
    
    
    
    



