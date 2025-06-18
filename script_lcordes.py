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

def test_models(pattern="", N=1000_000, dry=False):
    models_best = list(walk_dir(pattern + r".*model_best\.pt$"))
    preds_best = [max(walk_dir(f"tests/{torch.load(x, 'cpu').global_step}.*npz", x.parent))
                  for x in models_best]
    for model, pred in zip(models_best, preds_best):
        if str(N) not in str(pred):
            if dry: 
                print(model)
            else:
                model = torch.load(
                model, "cpu")
                model.test_model(N)
        

def main():    
    backbone = "/net/data_ttk/lcordes/ZToNuNu_copy/model_best.pt"
    src = Path("/net/data_ttk/lcordes/classifier_var_heads/QCD_vs_Aachen")
    
    model = torch.load(src / "NormalHead/model_last.pt")
    model.train_model(epochs=30, num_events=194350, num_events_test=64784, lr=2e-4, min_lr=2e-7)
    
    
    
if __name__=="__main__":
    t0 = time.time()
    main()
    print(f"--- finished in {datetime.timedelta(seconds=int(time.time() - t0))} ---")
