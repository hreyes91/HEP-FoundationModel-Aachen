from namespace import * 
torch.multiprocessing.set_sharing_strategy("file_system")

bf = lambda x: "$\\mathbf{" + x.replace(" ", "\\ ") + "}$"

class vdict(dict):
    def __getitem__(self, key):
        if isinstance(key, Iterable) and not isinstance(key, str):
            return np.array([dict.__getitem__(self, k) for k in key])
        return dict.__getitem__(self, key)


def reduce(x, length):
    assert len(x)>=length
    idx = np.linspace(0, len(x), length+1, dtype=int)
    out = [np.mean(x[idx[i]: idx[i+1]]) for i in range(len(idx)-1)]
    return np.array(out)


def walk_dir(pattern=r".*model_best\.pt$", dir=r"/net/data_ttk/lcordes/classifier_var_heads"):
    for dirpath, dirnames, filenames in os.walk(dir):
        for dirname in dirnames:
            if re.search(pattern, dirpath + "/" + dirname):
                yield Path(dirpath) / dirname
        for filename in filenames:
            if re.search(pattern, dirpath + "/" + filename):
                yield Path(dirpath) / filename
                
def select_max(dir, prefix=None):
    """selects the file or folder with the maximum global_step (non-recursive!)
    dir: looks here for files/dirs
    prefix: filters for paths with correct prefix. expected paths are: ".../<prefix>_<global_step>_..."
    """
    paths = list(Path(dir).iterdir())
    if prefix: 
        paths = [x for x in paths
                 if prefix==x.name.split("_")[0]]
    assert np.any(paths), f"No files/folders in dir matching prefix '{prefix}'\npaths are: {paths}"
    matches = [re.search(r"\d+", x.name) for x in paths]
    matches = [int(x.group()) if x else np.nan for x in matches]
    return paths[np.nanargmax(matches)]

def get_metrics(filenames, switch=False, signal_eff=0.3):
    """ returns aucs, bg_rejection, accuracy, epochs of best model
    """
    shape = np.shape(filenames)
    filenames = np.reshape(filenames, -1)
    
    epochs = []
    pred_files = []    
    for filename in filenames:
        model = torch.load(filename, "cpu")
        epochs.append(model.global_epoch)
        tests_folder = Path(filename).parent / f"tests/{model.global_step}_{model.global_epoch}"
        pred_files.append(select_max(tests_folder, "predictions"))
        
    aucs, bg_rejection, accuracy = [], [], []
    for pred_file in pred_files:
        data = np.load(pred_file)
        preds = (data["predictions"]).flatten()
        labels = data["labels"].flatten().astype(bool) ^ switch 
        accuracy.append(1 - np.abs(np.round(preds) - labels).sum() / len(labels))

        fpr, tpr, _ = sklearn.metrics.roc_curve(labels, preds)
        roc_auc = sklearn.metrics.auc(fpr, tpr)
        
        aucs.append(roc_auc)
        bg_rejection.append(1/fpr[np.argmin(np.abs(tpr - signal_eff))])
    
    return [np.reshape(aucs, shape), 
            np.reshape(bg_rejection, shape), 
            np.reshape(accuracy, shape),
            np.reshape(epochs, shape),
            ]
    
def select_max_auc(pattern="", dir="/net/data_ttk/lcordes/test_heads_and_protocols/Aachen/TTBar_Backbone"):
    pattern = pattern + ".*/model_best\.pt$"
    models = np.array(list(walk_dir(pattern=pattern, dir=dir)))
    aucs = get_metrics(models)[0]
    return models[np.argmax(aucs)]

def select_pred(model):
    model = torch.load(str(model), "cpu")
    return select_max(model.dir / "tests" / f"{model.global_step}_{model.global_epoch}")

def logger(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        sig = inspect.signature(f)
        bound = sig.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        
        log_data = {
            "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "params": {k: str(v) for k, v in bound.arguments.items()}
        }
        
        dir_path = Path(bound.arguments["dir"])
        dir_path.mkdir(exist_ok=True, parents=True)
        
        with open(dir_path / "args.jsonl", "a") as file:
            file.write(json.dumps(log_data) + "\n")
        
        return f(*args, **kwargs)
    return wrapper

class h:
    def bins2values(binned_data, bins):
        binned_data, bins = np.asarray(binned_data), np.asarray(bins).reshape(-1)
        vals = 1/2 * np.array([np.nan, *(bins[:-1] + bins[1:]), np.nan])
        return vals[binned_data.reshape(-1)].reshape(binned_data.shape)
    
    def denan(x):
        x = np.asarray(x)
        return x[x!=np.nan]
    
    def load_data(path, N):
        data = []
        for i,key in enumerate(["discretized", "pt_bins", "phi_bins", "eta_bins"]):
            x = pd.read_hdf(path, key=key, stop=N if i==0 else None).to_numpy()
            if i>0: x = x.flatten()
            data.append(x)
        return data[0][:,::3], data[0][:,1::3], data[0][:,2::3], *data[1:]
    
    def load_data2(path, N):
        data = np.asarray(pd.read_hdf(path, "discretized", stop=N))
        return [data[:,::3], data[:,1::3], data[:,2::3], 
                np.exp(np.load("/net/data_ttk/hreyes/OneBin/preprocessing_bins/pt_bins_1Mfromeach_403030.npy")),
                np.linspace(-.8,.8,30), np.linspace(-.8,.8,30)]
        
    def lim(x, offset=.05):
        l = np.array((np.nanmin(x), np.nanmax(x)))
        l += offset * np.diff(l) * [-1,1]
        return l 

    def info(filepaths, N=6):
        for filepath in np.atleast_1d(filepaths):
            size_MB = os.path.getsize(filepath) / 1024**2
            try:
                keys, shapes, columns, values = [], [], [], []
                store = pd.HDFStore(filepath, mode='r',)
                for key in store.keys():
                    df = store.get(key)
                    keys.append(key)
                    shapes.append(df.shape)
                    columns.append(list(df.columns))
                    if df.empty:
                        values.append(["<empty>"])
                    elif df.shape[1] == 1:
                        values.append(list(df.iloc[:min(N+1, df.shape[0]), 0]))
                    else:
                        values.append(list(df.iloc[0, :min(N+1, df.shape[1])]))
                store.close()
                
                columns = [" ".join(np.vectorize(str)([*(x[:N]), "..."] if len(x)>N else x)) for x in columns]
                values = [" ".join([*x[:N], "..."] if len(x)>N else x) for x in [vectorize(fn)(v) for v in values]]
                table([keys, shapes, columns, values],
                    ["keys", "shape", "columns", "values"],
                    True, caption=f"info for {filepath} ({fn(size_MB)} MB)")
            except:
                data = np.asarray(pd.read_hdf(filepath, key="table"))
                print(f"{color.fg.boldred(f'<info for {filepath} ({fn(size_MB)} MB)>')}\nshape of \\table: {data.shape}\n")
            
    def preprocess_data(sources, destination_folder, filename, split=[60,20,20], report=False):
        """ 
            filename: filename without ending (.h5)
        """
        def calculate_features(source, N=None):
            data = np.asarray(pd.read_hdf(source, key="table", stop=N))
            E, p_x, p_y, p_z = data[:,:-1:4], data[:,1::4], data[:,2::4], data[:,3::4]
            
            mask = E==0
            def nanify(x):
                x[mask] = np.nan 
                return x
                
            def f(E, p_x, p_y, p_z):
                p_T = np.sqrt(p_x**2 + p_y**2)
                phi = np.arctan2(p_y, p_x)
                eta = 1/2 * np.log((E + p_z) / (E - p_z))
                return p_T, phi, eta
            
            _, tot_phi, tot_eta = f(E.sum(1), p_x.sum(1), p_y.sum(1), p_z.sum(1))
            p_T, phi, eta = f(E, p_x, p_y, p_z)
            
            Delta_phi = (tot_phi[...,np.newaxis] - phi + np.pi) % (2*np.pi) - np.pi
            Delta_eta = tot_eta[...,np.newaxis] - eta
            
            return nanify(p_T), nanify(Delta_phi), nanify(Delta_eta)
        
        def get_bins(p_T, percent_inside):
            N = 100_000
            phi_eta_bins = np.linspace(-.8, .8, 30)
            p_T_bins = np.geomspace(np.nanpercentile(p_T[:N], 100-percent_inside),
                                    np.nanmax(p_T), 40) 
            # print(np.nanmin(p_T), np.nanmax(p_T), np.nanpercentile(p_T[:N], 100-percent_inside))
            return p_T_bins, phi_eta_bins, phi_eta_bins
        
        def discretize(x, bins):
            """ 
                x     = [pt, phi, eta]
                bins  = [pt_bins, phi_bins, eta_bins]
                returns [disk_pt, disk_phi, disk_eta]
            """
            mask = np.isnan(x[0])
            res = []
            for x,bins in zip(x,bins):
                x = np.digitize(x,bins).astype(np.int16)
                x[mask] = -1
                res.append(x)
            return np.array(res)
        
        def get_df(p_T, phi, eta):
            stacked = np.stack([p_T, eta, phi], -1,)
            stacked = stacked.reshape((-1, 600))
            cols = [
                item
                for sublist in [f"PT_{i},Eta_{i},Phi_{i}".split(",") for i in range(200)]
                for item in sublist
            ]
            df = pd.DataFrame(stacked, columns=cols)
            return df
        
        def save(disc_data, bins, split, destination_folder, filename): # disc_data = [p_T, phi, eta]
            path = Path(destination_folder)
            if not path.exists():
                os.mkdir(path)
                
            split_indices = np.cumsum(len(disc_data[0]) * np.asarray(split) / 100).astype(int)[:-1]
            split_data = np.split(disc_data, split_indices, axis=1)
            
            for data,label in zip(split_data, ["train", "val", "test"]):
                destination = path.joinpath(label + "_" + filename + ".h5")
                get_df(*data).to_hdf(destination, key="discretized", mode="w")
                
                for data,key in zip([*bins, split], ["pt_bins", "phi_bins", "eta_bins", "split"]):
                    pd.DataFrame(data).to_hdf(destination, key=key, mode="a")
                    
                if report: h.info(destination)
                    
            prep_dir = path.joinpath("preprocessing_bins")
            if not prep_dir.exists():
                os.mkdir(prep_dir)
            
            for bins, label in zip([*bins, split], ["pt_bins", "phi_bins", "eta_bins", "split"]):
                destination = prep_dir.joinpath(label + "_" + filename)
                np.save(destination, bins)
                
        features = np.hstack([calculate_features(source) for source in sources],)
        
        if report: print("\n"+color.fg.boldblue(f"<populating {destination_folder}, with {len(features[0])} jets total>"))
        
        bins = get_bins(features[0], 99.9)
        disc_data = discretize(features, bins)
        save(disc_data, bins, split, destination_folder, filename)
   
    paths = vdict({
        "ttbar/train": Path(r"/net/data_ttk/hreyes/OneBin/TTBar_train___1Mfromeach_403030.h5"),
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
        "qcd_aachen/val": Path(r"/net/data_ttk/lcordes/Semi-Visible/qcd_aachen_joined/val.h5"),
        
        "ttbar_classifier": Path(r"/net/data_ttk/lcordes/TTBar_ZToNuNu_classifier_1m_10e"),
        "aachen_classifier": Path(r"/net/data_ttk/lcordes/QCD_Aachen_classifier_1m_10e"),
        
        
        "ttbar_classifier_backbone": Path(r"/net/data_ttk/lcordes/TTBar_ZToNuNu_classifier_from_backbone_100k_10e"),
        # "aachen_classifier_backbone": Path(r"/net/data_ttk/lcordes/QCD_Aachen_classifier_1m_10e"),
    })
    
    def join_datasets(sources, destination_folder, split=(60,20,20)):
        df = pd.DataFrame()
        for path in sources:
            df = pd.concat([df, pd.read_hdf(path, key="discretized")])
        
        df = df.sample(frac=1).reset_index(drop=True)
        
        split_indices = np.cumsum(len(df) * np.asarray(split) / 100).astype(int)[:-1]
        dfs_split = np.split(df, split_indices)
        
        destination_folder = Path(destination_folder)
        destination_folder.mkdir(exist_ok=True)
        
        for df,label in zip(dfs_split, ["train", "val", "test"]):
            destination = destination_folder.joinpath(label + ".h5")
            df.to_hdf(destination, key="discretized", mode="w")


# Attention Rollout
from heads_lcordes import CLSTokenHead

def patch_transformer(transformer):
    def patch_attention(m):
        forward_orig = m.forward

        def wrap(*args, **kwargs):
            kwargs["need_weights"] = True
            kwargs["average_attn_weights"] = False

            return forward_orig(*args, **kwargs)

        m.forward = wrap

    class SaveOutput:
        def __init__(self):
            self.outputs = [] # [jet, heads, seq_len, seq_len]

        def __call__(self, module, module_in, module_out):
            self.outputs.append(module_out[1][0].detach())

        def clear(self):
            self.outputs = []
            
    save_outputs = []
    for i in range(len(transformer.layers)):
        save_outputs.append(SaveOutput())
        patch_attention(transformer.layers[i].self_attn)
        transformer.layers[i].self_attn.register_forward_hook(save_outputs[-1])
        
    if isinstance(transformer.head, CLSTokenHead):
        for i in range(len(transformer.head.transformer.layers)):
            save_outputs.append(SaveOutput())
            patch_attention(transformer.head.transformer.layers[i].self_attn)
            transformer.head.transformer.layers[i].self_attn.register_forward_hook(save_outputs[-1])
        
    return save_outputs

def get_attn(model, num_events = 100):
    """ attn.shape = [layer, jet, heads, seq_len, seq_len]
    """
    model = torch.load(model, "cpu")

    save_outputs = patch_transformer(model)
    jets, paddings, labels = [], [], []
    for jet, padding, label in tqdm(
        model.get_dataloader(num_events=num_events,batch_size=1,train=False,num_workers=0)
        ):
        jets.append(jet.numpy())
        paddings.append(padding.numpy())
        labels.append(label.numpy())
        _ = model(jet, padding)
        
    attn = [x.outputs for x in save_outputs]
    return attn, jets, paddings, labels

def augment_attn(attn, bb_layers=8):
    augmented_attn = np.zeros((len(attn), len(attn[0]), len(attn[0][0]), len(attn[0][0][0])+1, len(attn[0][0][0])+1))
    print(augmented_attn.shape)
    augmented_attn[..., 0, 0] = 1
    augmented_attn[:bb_layers, ..., 1:, 1:] = attn[:bb_layers]
    augmented_attn[bb_layers:, ...] = attn[bb_layers:]
    
    return augmented_attn

def get_attn_rollouts(attn, paddings, alpha=0.2):
    """ rollouts: [jet, seq_len, seq_len]
    """
    head_avg = attn.mean(2)   # [layers, jet, seq_len, seq_len]
    n_const = (~np.array(paddings)).sum(-1).sum(-1)

    rollouts = []
    for b in range(head_avg.shape[1]):
        rollout = np.eye(head_avg.shape[-1])
        for l in range(head_avg.shape[0]):
            A = head_avg[l, b]
            A_aug = alpha * A + (1 - alpha) * np.eye(A.shape[0])
            A_aug = A_aug / A_aug.sum(-1, keepdims=True)
            rollout = A_aug @ rollout
        rollouts.append(rollout)

    rollouts = np.stack(rollouts, axis=0)  
    return rollouts # [jet, seq_len, seq_len]

def gensave_rollouts(model, num_events=10_000, alpha=0.2):
    """ saved as: <model_dir>/tests/<global_step>_<global_epoch>/rollouts_<alpha>_<num_events>.npz
    """
    attn, jets, paddings, labels = get_attn(model, num_events)
    aug_attn = augment_attn(attn, bb_layers=8)
    rollouts = get_attn_rollouts(aug_attn, paddings, alpha)
    
    m = torch.load(model, "cpu")
    filename = m.dir / "tests" / f"{m.global_step}_{m.global_epoch}" / f"rollouts_{alpha}_{num_events}"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    np.savez(filename, rollouts=rollouts, aug_attn=aug_attn, jets=jets, paddings=paddings, labels=labels)
    print(f"Rollouts saved as: '{filename}.npz'")
    
    return f"{filename}.npz"