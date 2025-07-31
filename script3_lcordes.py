from model_lcordes import *
# os.chdir("/net/data_ttk/lcordes/classifier_var_heads")

paths = {"ttbar/train": Path(r"/net/data_ttk/hreyes/OneBin/TTBar_train___1Mfromeach_403030.h5"),
        "ttbar/test": Path(r"/net/data_ttk/hreyes/OneBin/TTBar_test___1Mfromeach_403030.h5"),
        "ttbar/val": Path(r"/net/data_ttk/hreyes/OneBin/TTBar_val___1Mfromeach_403030.h5"),
        "ttbar/samples": Path(r"/net/data_ttk/lcordes/TTBar_500k/samples_100k.h5"),
        
        "z/train": Path(r"/net/data_ttk/hreyes/OneBin/ZJetsToNuNu_train___1Mfromeach_403030.h5"),
        "z/test": Path(r"/net/data_ttk/hreyes/OneBin/ZJetsToNuNu_test___1Mfromeach_403030.h5"),
        "z/val": Path(r"/net/data_ttk/hreyes/OneBin/ZJetsToNuNu_val___1Mfromeach_403030.h5"),
        
        "qcd": Path(r"/net/data_ttk/lcordes/redo/semi-visible/qcd"),
        "qcd/train": Path(r"/net/data_ttk/lcordes/redo/semi-visible/qcd/train_qcd.h5"),
        "qcd/test": Path(r"/net/data_ttk/lcordes/redo/semi-visible/qcd/test_qcd.h5"),
        "qcd/val": Path(r"/net/data_ttk/lcordes/redo/semi-visible/qcd/val_qcd.h5"),
        
        "aachen": Path(r"/net/data_ttk/lcordes/redo/semi-visible/aachen"),
        "aachen/train": Path(r"/net/data_ttk/lcordes/redo/semi-visible/aachen/train_aachen.h5"),
        "aachen/test": Path(r"/net/data_ttk/lcordes/redo/semi-visible/aachen/test_aachen.h5"),
        "aachen/val": Path(r"/net/data_ttk/lcordes/redo/semi-visible/aachen/val_aachen.h5")
        }      


def test_heads_and_protocols(backbone, src_dir):    
    head_labels = ["Linear", "MaxPool", "MeanPool", "AttnPool", "CLSToken"]
    heads = [NormalHead, MaxPoolHead, MeanPoolHead, SimpleAttentionPoolHead, CLSTokenHead]
    
    bg_file = paths["qcd/train"]
    sig_file = paths["aachen/train"]
    num_events = 151175
    num_test_events = 50392
    
    
    for append_to_model_dir in ["_1", "_2", "_3"]:
        
        # Freeze_ReducedLR
        for head, head_label in zip(heads, head_labels):
            model_dir = Path(src_dir) / f"{head_label}_Freeze_Reduced_LR{append_to_model_dir}"
            
            model = JetClassifier2(
                dir=model_dir,
                backbone=backbone,
                head=head(),
                bg_file=bg_file,
                sig_file=sig_file,
                )
            
            model.train_model(
                epochs=1,
                num_events=num_events,
                lr=1e-4, 
                min_lr=1e-6, 
                weight_decay=1e-5, 
                batch_size=100, 
                num_workers=1, 
                use_profiler=False, 
                num_events_test=0, 
                testing_steps=100, 
                logging_steps=100, 
                dropout_p=0.1, 
                checkpoint=True, 
                freeze=True, 
                scheduler="Constant"
                )
            
            model.train_model(
                epochs=9,
                num_events=num_events,
                lr=1e-4, 
                min_lr=1e-6, 
                weight_decay=1e-5, 
                batch_size=100, 
                num_workers=1, 
                use_profiler=False, 
                num_events_test=0, 
                testing_steps=1, 
                logging_steps=100, 
                dropout_p=0.1, 
                checkpoint=True, 
                freeze=False, 
                scheduler="CosineAnnealingLR"
                )
            
            model = torch.load(model_dir / "model_best.pt", "cpu")
            model.test_model(num_events=num_test_events, log=True)
    
        # From Scratch
        for head, head_label in zip(heads, head_labels):
            model_dir = Path(src_dir) / f"{head_label}_Scratch{append_to_model_dir}"
            
            model = JetClassifier2(
                dir=model_dir,
                backbone=JetTransformer(),
                head=head(),
                bg_file=bg_file,
                sig_file=sig_file,
                )
            
            model.train_model(
                epochs=10,
                num_events=num_events,
                lr=1e-3, 
                min_lr=1e-6, 
                weight_decay=1e-5, 
                batch_size=100, 
                num_workers=1, 
                use_profiler=False, 
                num_events_test=0, 
                testing_steps=1, 
                logging_steps=100, 
                dropout_p=0.1, 
                checkpoint=True, 
                freeze=False, 
                scheduler="CosineAnnealingLR"
                )
            
            model = torch.load(model_dir / "model_best.pt", "cpu")
            model.test_model(num_events=num_test_events, log=True)
            

        # Baseline
        for head, head_label in zip(heads, head_labels):
            model_dir = Path(src_dir) / f"{head_label}_Baseline{append_to_model_dir}"
            
            model = JetClassifier2(
                dir=model_dir,
                backbone=backbone,
                head=head(),
                bg_file=bg_file,
                sig_file=sig_file,
                )
            
            model.train_model(
                epochs=10,
                num_events=num_events,
                lr=1e-3, 
                min_lr=1e-6, 
                weight_decay=1e-5, 
                batch_size=100, 
                num_workers=1, 
                use_profiler=False, 
                num_events_test=0, 
                testing_steps=1, 
                logging_steps=100, 
                dropout_p=0.1, 
                checkpoint=True, 
                freeze=False, 
                scheduler="CosineAnnealingLR"
                )
            
            model = torch.load(model_dir / "model_best.pt", "cpu")
            model.test_model(num_events=num_test_events, log=True)
            
        
        # Reduced LR
        for head, head_label in zip(heads, head_labels):
            model_dir = Path(src_dir) / f"{head_label}_ReducedLR{append_to_model_dir}"
            
            model = JetClassifier2(
                dir=model_dir,
                backbone=backbone,
                head=head(),
                bg_file=bg_file,
                sig_file=sig_file,
                )
            
            model.train_model(
                epochs=10,
                num_events=num_events,
                lr=1e-4, 
                min_lr=1e-6, 
                weight_decay=1e-5, 
                batch_size=100, 
                num_workers=1, 
                use_profiler=False, 
                num_events_test=0, 
                testing_steps=1, 
                logging_steps=100, 
                dropout_p=0.1, 
                checkpoint=True, 
                freeze=False, 
                scheduler="CosineAnnealingLR"
                )
            
            model = torch.load(model_dir / "model_best.pt", "cpu")
            model.test_model(num_events=num_test_events, log=True)
            
        
        # Freeze
        for head, head_label in zip(heads, head_labels):
            model_dir = Path(src_dir) / f"{head_label}_Freeze{append_to_model_dir}"
            
            model = JetClassifier2(
                dir=model_dir,
                backbone=backbone,
                head=head(),
                bg_file=bg_file,
                sig_file=sig_file,
                )
            
            model.train_model(
                epochs=1,
                num_events=num_events,
                lr=1e-3, 
                min_lr=1e-6, 
                weight_decay=1e-5, 
                batch_size=100, 
                num_workers=1, 
                use_profiler=False, 
                num_events_test=0, 
                testing_steps=1, 
                logging_steps=100, 
                dropout_p=0.1, 
                checkpoint=False, 
                freeze=True, 
                scheduler="Constant"
                )
            
            model.train_model(
                epochs=9,
                num_events=num_events,
                lr=1e-3, 
                min_lr=1e-6, 
                weight_decay=1e-5, 
                batch_size=100, 
                num_workers=1, 
                use_profiler=False, 
                num_events_test=0, 
                testing_steps=1, 
                logging_steps=100, 
                dropout_p=0.1, 
                checkpoint=True, 
                freeze=False, 
                scheduler="CosineAnnealingLR"
                )
            
            model = torch.load(model_dir / "model_best.pt", "cpu")
            model.test_model(num_events=num_test_events, log=True)
        
  
def main():    
    # train Aachen classifiers from TTbar backbones
    TTBar_backbone = "/net/data_ttk/lcordes/TTBar_10m_20e/model_best.pt"
    Z_backbone = "/net/data_ttk/lcordes/ZToNuNu_10m_20e/model_best.pt"
    test_heads_and_protocols(TTBar_backbone,
                             "/net/data_ttk/lcordes/redo/classifiers/ttbar_backbone")

if __name__=="__main__":
    t0 = time.time()
    main()
    print(f"--- finished in {datetime.timedelta(seconds=int(time.time() - t0))} ---")
