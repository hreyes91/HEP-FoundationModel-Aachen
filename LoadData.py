from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from argparse import ArgumentParser

f#rom model_finetune import JetTransformerClassifierFine

from tqdm import tqdm
import pandas as pd
import os

from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt



def load_data(file):

    jet1 = pd.read_hdf(file, key="discretized", stop=args.num_events)
    jet1 = jet1.to_numpy(dtype=np.int64)[:, : args.num_const * 3]
    jet1 = jet1.reshape(dat.shape[0], -1, 3)

    jet1 = np.delete(jet1, np.where(dat[:, 0, 0] == 0)[0], axis=0)
    jet1[jet1 == -1] = 0
    
    jet2 = pd.read_hdf(file, key="discretized", stop=args.num_events)
    jet2 = jet2.to_numpy(dtype=np.int64)[:, : args.num_const * 3]
    jet2 = jet2.reshape(dat.shape[0], -1, 3)

    jet2 = np.delete(jet2, np.where(dat[:, 0, 0] == 0)[0], axis=0)
    jet2[jet2 == -1] = 0
    

    
    return jet1,jet2



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

    print(f"Using bg {bg.shape} from {bgf} and sig {sig.shape} from {sigf}")

    jet1_data = np.concatenate((bg_jet1, sig_jet1), 0)
    jet2_data = np.concatenate((bg_jet2, sig_jet2), 0)
    
    
    label = np.append(np.zeros(len(bg_jet1)), np.ones(len(sig_jet1)))
    
    padding_mask1 = jet1_data[:, :, 0] != 0
    padding_mask2 = jet2_data[:, :, 0] != 0





    idx = np.random.permutation(len(label))
    
    jet1 = torch.tensor(jet1_data[idx], dtype=torch.float32)
    jet2 = torch.tensor(jet2_data[idx], dtype=torch.float32)
    padding_mask1 = torch.tensor(padding_mask1[idx], dtype=torch.bool)
    padding_mask2 = torch.tensor(padding_mask2[idx], dtype=torch.bool)
    label = torch.tensor(self.labels[idx], dtype=torch.float32)  # For BCEWithLogitsLoss, labels should be float32
    
    
    train_set = TensorDataset(
        jet1[: int(0.9 * len(dat))],
        padding_mask1[: int(0.9 * len(dat))],
        jet2[: int(0.9 * len(dat))],
        padding_mask2[: int(0.9 * len(dat))],
        
        
        label[: int(0.9 * len(dat))],
    )
    
    
    val_set = TensorDataset(
        jet1[int(0.9 * len(dat)) :],
        padding_mask1[int(0.9 * len(dat)) :],
        jet2[int(0.9 * len(dat)) :],
        padding_mask2[int(0.9 * len(dat)) :],
        
        
        label[int(0.9 * len(dat)) :],
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
    
    # Create the DataLoader
    #dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    
    return train_loader,val_loader
