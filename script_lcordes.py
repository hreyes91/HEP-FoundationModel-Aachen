from model_lcordes import *

paths = {"ttbar/train": Path(r"/net/data_ttk/hreyes/OneBin/TTBar_train___1Mfromeach_403030.h5"),
        "ttbar/test": Path(r"/net/data_ttk/hreyes/OneBin/TTBar_test___1Mfromeach_403030.h5"),
        "ttbar/val": Path(r"/net/data_ttk/hreyes/OneBin/TTBar_val___1Mfromeach_403030.h5"),
        "ttbar/samples": Path(r"/net/data_ttk/lcordes/TTBar_500k/samples_100k.h5"),
        
        "z/train": Path(r"/net/data_ttk/hreyes/OneBin/ZJetsToNuNu_train___1Mfromeach_403030.h5"),
        "z/test": Path(r"/net/data_ttk/hreyes/OneBin/ZJetsToNuNu_test___1Mfromeach_403030.h5"),
        "z/val": Path(r"/net/data_ttk/hreyes/OneBin/ZJetsToNuNu_val___1Mfromeach_403030.h5"),
        
        "qcd": Path(r"/net/data_ttk/lcordes/Semi-Visible/qcd/"),
        "qcd/train": Path(r"/net/data_ttk/lcordes/Semi-Visible/qcd/train_qcd_disc.h5"),
        "qcd/test": Path(r"/net/data_ttk/lcordes/Semi-Visible/qcd/test_qcd_disc.h5"),
        "qcd/val": Path(r"/net/data_ttk/lcordes/Semi-Visible/qcd/val_qcd_disc.h5"),
        
        "aachen": Path(r"/net/data_ttk/lcordes/Semi-Visible/aachen/"),
        "aachen/train": Path(r"/net/data_ttk/lcordes/Semi-Visible/aachen/train_aachen_disc.h5"),
        "aachen/test": Path(r"/net/data_ttk/lcordes/Semi-Visible/aachen/test_aachen_disc.h5"),
        "aachen/val": Path(r"/net/data_ttk/lcordes/Semi-Visible/aachen/val_aachen_disc.h5"),
        
        "qcd_aachen": Path(r"/net/data_ttk/lcordes/Semi-Visible/qcd_aachen_joined/"),
        "qcd_aachen/train": Path(r"/net/data_ttk/lcordes/Semi-Visible/qcd_aachen_joined/train.h5"),
        "qcd_aachen/test": Path(r"/net/data_ttk/lcordes/Semi-Visible/qcd_aachen_joined/test.h5"),
        "qcd_aachen/val": Path(r"/net/data_ttk/lcordes/Semi-Visible/qcd_aachen_joined/val.h5")}

def test(pattern="", N=1000_000):
    models_best = list(walk_dir(pattern + r".*model_best\.pt$"))
    preds_best = [max(walk_dir(f"tests/{torch.load(x, 'cpu').global_step}.*npz", x.parent))
                  for x in models_best]
    for model, pred in zip(models_best, preds_best):
        if str(N) not in str(pred):
            model = torch.load(model, "cpu")
            model.test_model(N)
            

def main():
    test()
    # model = JetClassifier(dir = r"/net/data_ttk/lcordes/classifier_var_heads/QCD_vs_Top/MeanPoolHead",
    #                       backbone = r"/net/data_ttk/lcordes/TTBar_10m_20e/model_best.pt",
    #                       head = MeanPoolHead(), 
    #                       bg_file = paths["z/train"],
    #                       sig_file = paths["ttbar/train"])
    
    # model.train_model()
    
    
    # model = JetClassifier(dir = r"/net/data_ttk/lcordes/classifier_var_heads/QCD_vs_Top/MaxPoolHead",
    #                       backbone = r"/net/data_ttk/lcordes/TTBar_10m_20e/model_best.pt",
    #                       head = MaxPoolHead(), 
    #                       bg_file = paths["z/train"],
    #                       sig_file = paths["ttbar/train"])
    
    # model.train_model()
    
    
if __name__=="__main__":
    t0 = time.time()
    main()
    print(f"--- finished in {datetime.timedelta(seconds=int(time.time() - t0))} ---")
