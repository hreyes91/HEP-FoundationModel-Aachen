import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import roc_auc_score

from preprocess import preprocess_dataframe

from argparse import ArgumentParser
from tqdm import tqdm
import os
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

torch.multiprocessing.set_sharing_strategy('file_system')






def preprocess_dataframe(
    df,
    num_features,
    num_bins,
    num_const,
    to_tensor=True,
    reverse=False,
    start=False,
    end=False,
    limit_nconst=False,
):
    x = df.to_numpy(dtype=np.int64)[:, : num_const * num_features]
    
    
    
    #x = x.reshape(x.shape[0], -1, num_features)
    padding_mask = x[:, :, 0] != -1

    if limit_nconst:
        keepings = padding_mask.sum(-1) >= num_const
        x = x[keepings]
        padding_mask = padding_mask[keepings]

    if reverse:
        print("Reversing pt order")
        x[x == -1] = np.max(num_bins) + 10
        idx_sort = np.argsort(x[:, :, 0], axis=-1)
        for i in range(len(x)):
            x[i] = x[i, idx_sort[i]]
        x[x == np.max(num_bins) + 10] = -1

    num_prior_bins = np.cumprod((1,) + num_bins[:-1])
    bins = (x * num_prior_bins.reshape(1, 1, num_features)).sum(axis=2)

    if start:
        print("Adding start particles")
        bins = np.concatenate(
            (np.ones((len(bins), 1), dtype=int) * -100, bins),
            axis=1,
        )

        x = np.concatenate(
            (
                np.zeros((len(x), 1, num_features), dtype=int),
                x,
            ),
            axis=1,
        )
        padding_mask = x[:, :, 0] != -1
        bins[~padding_mask] = -100
    else:
        bins[~padding_mask] = -100

    if end:
        print("Adding stop token")
        seq_lengths = padding_mask.sum(-1)
        x = np.append(x, -np.ones((x.shape[0], 1, x.shape[2]), dtype=int), axis=1)
        x[np.arange(x.shape[0]), seq_lengths] = 0
        x = x[:, :-1]
        bins = np.append(bins, -100 * np.ones((bins.shape[0], 1)).astype(int), axis=1)
        bins[np.arange(bins.shape[0]), seq_lengths] = np.prod(num_bins)
        bins = bins[:, :-1]
        padding_mask = x[:, :, 0] != -1

    if to_tensor:
        x = torch.tensor(x)
        padding_mask = torch.tensor(padding_mask)
        bins = torch.tensor(bins)
    print(f"Shapes: {x.shape=} {padding_mask.shape=} {bins.shape=}")
    return x, padding_mask, bins
'''
def load_data(file):

    jet1 = pd.read_hdf(file, key="discretized_jet1", stop=args.num_events)
    jet1 = jet1.to_numpy(dtype=np.int64)[:, : args.num_const * 3]
    jet1 = jet1.reshape(jet1.shape[0], -1, 3)

    jet1 = np.delete(jet1, np.where(jet1[:, 0, 0] == 0)[0], axis=0)
    jet1[jet1 == -1] = 0
    
    jet2 = pd.read_hdf(file, key="discretized_jet2", stop=args.num_events)
    jet2 = jet2.to_numpy(dtype=np.int64)[:, : args.num_const * 3]
    jet2 = jet2.reshape(jet2.shape[0], -1, 3)

    jet2 = np.delete(jet2, np.where(jet2[:, 0, 0] == 0)[0], axis=0)
    jet2[jet2 == -1] = 0
    

    
    return jet1,jet2
'''
def load_data(file):

    jet1 = pd.read_hdf(file, key="discretized_jet1", stop=args.num_events)
    jet1 = jet1.to_numpy(dtype=np.int64)[:, : args.num_const * 3]
    jet1 = jet1.reshape(jet1.shape[0], -1, 3)

    jet1 = np.delete(jet1, np.where(jet1[:, 0, 0] == 0)[0], axis=0)
    jet1[jet1 == -1] = 0
    
    jet2 = pd.read_hdf(file, key="discretized_jet2", stop=args.num_events)
    jet2 = jet2.to_numpy(dtype=np.int64)[:, : args.num_const * 3]
    jet2 = jet2.reshape(jet2.shape[0], -1, 3)

    jet2 = np.delete(jet2, np.where(jet2[:, 0, 0] == 0)[0], axis=0)
    jet2[jet2 == -1] = 0
    
    f=h5py.File(file, 'r')
    jet_coords=f.get('jet_coords')[:args.num_events,:,:]
    
    return jet1,jet2,jet_coords

'''
def get_dataloader(bgf,sigf, batch_size=32, shuffle=False, num_workers=4):
    """
    Creates a DataLoader for training or validation.
    
    Args:
        jet1_data (numpy.ndarray or torch.Tensor): Jet 1 data.
        jet2_data (numpy.ndarray or torch.Tensor): Jet 2 data.
        padding_mask1 (numpy.ndarray or torch.Tensor): Padding mask for Jet 1.
        padding_mask2 (numpy.ndarray or torch.Tensor): Padding mask for Jet 2.
        labels (numpy.ndarray or torch.Tensor): Labels for binary classification.
        batch_size (int): The size of each batch (default is 32).
        shuffle (bool): Whether to shuffle the data (default is True).
        num_workers (int): The number of workers to load data (default is 4).
    
    Returns:
        DataLoader: The DataLoader object for training or validation.
    """
    
    
    bg_jet1,bg_jet2 = load_data(bgf)
    sig_jet1,sig_jet2 = load_data(sigf)

    print(f"Using bg {bg_jet1.shape} from {bg_jet1} and sig {sig_jet1.shape} from {sig_jet1}")

    jet1_data = np.concatenate((bg_jet1, sig_jet1), 0)
    jet2_data = np.concatenate((bg_jet2, sig_jet2), 0)
    
    
    label = np.append(np.zeros(len(bg_jet1)), np.ones(len(sig_jet1)))
    
    padding_mask1 = jet1_data[:, :, 0] != 0
    padding_mask2 = jet2_data[:, :, 0] != 0





    #idx = np.random.permutation(len(label))
    
    jet1 = torch.tensor(jet1_data, dtype=torch.int64)
    jet2 = torch.tensor(jet2_data, dtype=torch.int64)
    padding_mask1 = torch.tensor(padding_mask1, dtype=torch.bool)
    padding_mask2 = torch.tensor(padding_mask2, dtype=torch.bool)
    label = torch.tensor(label, dtype=torch.float32)  # For BCEWithLogitsLoss, labels should be float32
    
    
    test_set = TensorDataset(
        jet1[: ],
        padding_mask1[:],
        jet2[: ],
        padding_mask2[: ],
        
        
        label[: ],
    )
    

    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=shuffle,
    )

    return test_loader
'''


def get_dataloader(bgf,sigf, batch_size=32, shuffle=False, num_workers=4):
    """
    Creates a DataLoader for training or validation.
    
    Args:
        jet1_data (numpy.ndarray or torch.Tensor): Jet 1 data.
        jet2_data (numpy.ndarray or torch.Tensor): Jet 2 data.
        padding_mask1 (numpy.ndarray or torch.Tensor): Padding mask for Jet 1.
        padding_mask2 (numpy.ndarray or torch.Tensor): Padding mask for Jet 2.
        labels (numpy.ndarray or torch.Tensor): Labels for binary classification.
        batch_size (int): The size of each batch (default is 32).
        shuffle (bool): Whether to shuffle the data (default is True).
        num_workers (int): The number of workers to load data (default is 4).
    
    Returns:
        DataLoader: The DataLoader object for training or validation.
    """
    
    
    bg_jet1,bg_jet2,bg_jet_coords = load_data(bgf)
    sig_jet1,sig_jet2,sg_jet_coords = load_data(sigf)

    print(f"Using bg {bg_jet1.shape} from {bg_jet1} and sig {sig_jet1.shape} from {sig_jet1}")

    jet1_data = np.concatenate((bg_jet1, sig_jet1), 0)
    jet2_data = np.concatenate((bg_jet2, sig_jet2), 0)
    jet_coords_data=np.concatenate((bg_jet_coords, sg_jet_coords), 0)
    
    label = np.append(np.zeros(len(bg_jet1)), np.ones(len(sig_jet1)))
    
    padding_mask1 = jet1_data[:, :, 0] != 0
    padding_mask2 = jet2_data[:, :, 0] != 0





    #idx = np.random.permutation(len(label))
    
    jet1 = torch.tensor(jet1_data, dtype=torch.int64)
    jet2 = torch.tensor(jet2_data, dtype=torch.int64)
    padding_mask1 = torch.tensor(padding_mask1, dtype=torch.bool)
    padding_mask2 = torch.tensor(padding_mask2, dtype=torch.bool)
    label = torch.tensor(label, dtype=torch.float32)  # For BCEWithLogitsLoss, labels should be float32
    jet_coords = torch.tensor(jet_coords_data, dtype=torch.float32)
    
    print('jet1')
    print(jet1.shape)
    print('jet2')
    print(jet2.shape)
    print('padding_mask1')
    print(padding_mask1.shape)
    print('padding_mask2')
    print(padding_mask2.shape)
    print('jet_coords')
    print(jet_coords.shape)
    
    
    
    test_set = TensorDataset(
        jet1,
        padding_mask1,
        jet2,
        padding_mask2,
        jet_coords,
        label,
    )
    

    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=shuffle,
    )

    return test_loader


def load_model(name):
    model = torch.load(os.path.join(args.model_dir, f'model_{name}.pt'))
    return model


def parse_input():
    parser = ArgumentParser()
    parser.add_argument("--model_dir", type=str, default='models/test', help="Model directory")
    parser.add_argument("--data_path_1", type=str, default='/hpcwork/bn227573/top_benchmark/', help="Path to training data file")
    parser.add_argument("--data_path_2", type=str, default='/hpcwork/bn227573/qcd_benchmark/', help="Path to training data file")

    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers")
    parser.add_argument("--batch_size", type=int, default=128, help="Number of workers")

    parser.add_argument("--num_const", type=int, default=100, help="Number of constituents")
    parser.add_argument("--limit_const", action="store_true", help="Only use jets with at least num_const constituents")
    parser.add_argument("--num_events", type=int, default=10000, help="Number of events for training")
    parser.add_argument("--num_bins", type=int, nargs=3, default=[41, 31, 31], help="Number of bins per feature")
    parser.add_argument("--reverse", action='store_true', help="Whether to reverse pt order")
    parser.add_argument("--model_name", type=str, default='best', help="model name")
    parser.add_argument("--pred_name", type=str, default='predictions_test.npz', help="predicitons name")
    parser.add_argument("--use_hlf", type=str, default='False', help="use hlf info")
    args = parser.parse_args()
    return args



    
    
    




def plot_roc_curve(y_true, y_score,model_dir):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure()
    lw = 2
    plt.plot(tpr,1/fpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    #plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

    plt.xlabel('True Positive Rate')
    plt.ylabel('False Positive Rate')
    plt.yscale('log')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(model_dir+'/roc_test_1fpr.png')
    plt.close()
    
    
    
    sic_values = np.where(fpr > 0, tpr / np.sqrt(fpr), 0)

    # Plot SIC curve
    plt.figure(figsize=(8, 6))
    plt.plot(tpr, sic_values, label='SIC Curve', color='b', lw=2)
    plt.xlabel("Signal Efficiency (εS)")
    plt.ylabel("Significance Improvement Factor (SIF)")
    plt.title("Significance Improvement Characteristic (SIC) Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(model_dir+'/sic_curve.png')
    
    return






def saveAUCscore(model_dir,auc_score):

    file=open(model_dir+'/auc.txt','w')
    file.write('auc_score')
    file.write('\n')
    file.write(str(auc_score))
    file.write('\n')
    file.close()

    return

if __name__ == '__main__':
    args = parse_input()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on device: {device}")

    num_features = 3
    num_bins = tuple(args.num_bins)

    print(f"Using bins: {num_bins}")
    print(f"{'Not' if not args.reverse else ''} reversing pt order")

    # load and preprocess data
    print(f"Loading test set")
    test_loader = get_dataloader(args.data_path_1,args.data_path_2)
    # construct model
    model = load_model(args.model_name)

    if args.model_name=='ensemble':

        model_1 = load_model('best')
        model_2 = load_model('last')
        model_3 = load_model('best_train')
        model_4 = load_model('epoch_3')
        model_5 = load_model('epoch_8')
        model_list=[model_1,model_2,model_3,model_4,model_5]

    print("Loaded model")
    model.to(device)
    model.eval()

    loss_list = []
    prediction_list = []
    label_list = []
    logits_list=[]
    min_val_loss = np.inf

    if args.model_name == 'ensemble':
        
        for model in model_list:


            with torch.no_grad():
                for jet1, padding_mask1,jet2, padding_mask2, hlf,label in tqdm(test_loader, total=len(test_loader), desc=f'Testing'):
                    label_list.append(label.detach().numpy())
                
                    jet1 = jet1.to(device)
                    padding_mask1 = padding_mask1.to(device)
                
                    jet2 = jet2.to(device)
                    padding_mask2 = padding_mask2.to(device)
                
                    label = label.to(device)
                    hlf=hlf.to(device)
                    #with torch.no_grad():
                    #with torch.cuda.amp.autocast():
                    logits = model(jet1, jet2, padding_mask1, padding_mask2,hlf)
                    predictions = torch.sigmoid(logits)
                    loss = model.loss(logits, label.view(-1, 1))

                    loss_list.append(loss.cpu().detach().numpy())
                    prediction_list.append(predictions.cpu().detach().numpy())
                    logits_list.append(logits.cpu().detach().numpy())

            predictions = np.concatenate(prediction_list, axis=0)
            logits_all=np.concatenate(logits_list, axis=0)
            label_all = np.concatenate(label_list, axis=0)

            predictions=predictions[:,0]


            

    else:
        with torch.no_grad():
            for jet1, padding_mask1,jet2, padding_mask2,hlf, label in tqdm(test_loader, total=len(test_loader), desc=f'Testing'):
                label_list.append(label.detach().numpy())
                
                jet1 = jet1.to(device)
                padding_mask1 = padding_mask1.to(device)
                
                jet2 = jet2.to(device)
                padding_mask2 = padding_mask2.to(device)
                
                label = label.to(device)
                hlf=hlf.to(device)
                #with torch.no_grad():
                #with torch.cuda.amp.autocast():
                logits = model(jet1, jet2, padding_mask1, padding_mask2,hlf)
                predictions = torch.sigmoid(logits)
                loss = model.loss(logits, label.view(-1, 1))

                loss_list.append(loss.cpu().detach().numpy())
                prediction_list.append(predictions.cpu().detach().numpy())
                logits_list.append(logits.cpu().detach().numpy())

        predictions = np.concatenate(prediction_list, axis=0)
        logits_all=np.concatenate(logits_list, axis=0)
        label_all = np.concatenate(label_list, axis=0)

        predictions=predictions[:,0]



    auc_score=roc_auc_score(label_all, predictions)
    print(auc_score)
    
    plot_roc_curve(label_all, predictions,args.model_dir)
    saveAUCscore(args.model_dir,auc_score)
    
    np.savez(os.path.join(args.model_dir, args.pred_name),
            predictions=predictions,
            labels=label_all,
            logits=logits_all)
