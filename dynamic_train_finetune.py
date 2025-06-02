import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from argparse import ArgumentParser

from model_finetune import JetTransformerClassifierFine

from tqdm import tqdm
import pandas as pd
import os

from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

from helpers_train import (
    get_cos_scheduler,
    save_opt_states,
    save_model,
    save_arguments,
    set_seeds,
    load_model,
)

torch.multiprocessing.set_sharing_strategy("file_system")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def parse_input():
    parser = ArgumentParser()
    parser.add_argument("--log_dir", 
                        type=str, 
                        default="models/test", 
                        help="Model directory")
    parser.add_argument("--bg", 
                        type=str, 
                        default="/hpcwork/bn227573/top_benchmark/train_qcd_30_bins.h5", 
                        help="Path to background data file")
    parser.add_argument("--sig", 
                        type=str, 
                        default="/hpcwork/bn227573/top_benchmark/train_top_30_bins.h5", 
                        help="Path to signal data file")
    parser.add_argument("--seed", 
                        type=int, 
                        default=0, 
                        help="the random seed for torch and numpy")
    parser.add_argument("--logging_steps", 
                        type=int, 
                        default=10, 
                        help="Training steps between logging")
    parser.add_argument("--model_path_in", 
                        type=str, 
                        default="model/test_in", 
                        help="Pretrained model")
    parser.add_argument("--model_name", 
                        type=str, 
                        default="model_best.pt", 
                        help="Pretrained model choice")
    parser.add_argument("--num_events_val", 
                        type=int, 
                        default=10000, 
                        help="Number of val events for training")
    parser.add_argument("--num_workers", 
                        type=int, 
                        default=4, 
                        help="Number of workers")
    parser.add_argument("--num_const", 
                        type=int, 
                        default=128, 
                        help="Number of constituents")
    parser.add_argument("--num_events", 
                        type=int, 
                        default=10000, 
                        help="Number of events for training")
    parser.add_argument("--num_bins", 
                        type=int, 
                        nargs=3, 
                        default=[41, 31, 31], 
                        help="Number of bins per feature")
    parser.add_argument("--name_sufix", 
                        type=str, 
                        default="A1B2C3D", 
                        help="name of train dir")
    parser.add_argument("--num_epochs", 
                        type=int, 
                        default=3, 
                        help="Number of epochs")
    parser.add_argument("--batch_size", 
                        type=int, 
                        default=100, 
                        help="Batch size")
    parser.add_argument("--lr", 
                        type=float, 
                        default=0.001, 
                        help="learning rate")
    parser.add_argument("--weight_decay", 
                        type=float, default=0.00001, 
                        help="weight decay")
    parser.add_argument("--hidden_dim", 
                        type=int, 
                        default=256, 
                        help="Hidden dim of the model")
    parser.add_argument("--num_layers", 
                        type=int, 
                        default=8, 
                        help="Number of transformer layers")
    parser.add_argument("--num_heads", 
                        type=int, 
                        default=4, 
                        help="Number of attention heads")
    parser.add_argument("--dropout", 
                        type=float, 
                        default=0.1, 
                        help="dropout rate")
    parser.add_argument("--output", 
                        type=str, 
                        default="linear", 
                        choices=["linear", "embprod"], help="Output function")
    parser.add_argument("--learnrate_factor",
                        type="float",
                        default=0.5,
                        help="multiply learnrate by <learnrate_factor> after <patience> epochs without improvement")
    parser.add_argument("--patience", 
                        type=int,
                        default=2,
                        help="multiply learnrate by <learnrate_factor> after <patience> epochs without improvement")
    parser.add_argument("--max_patience", 
                        type=int, 
                        default=5, 
                        help="stop training prematurely after <max_patience> epochs without improvement") # prev early_stop_patience
    args = parser.parse_args()
    return args


def load_data(file):
    if file.endswith("npz"):
        dat = np.load(file)["jets"][: args.num_events, : args.num_const]
    elif file.endswith("h5"):
        dat = pd.read_hdf(file, key="discretized", stop=args.num_events)
        dat = dat.to_numpy(dtype=np.int64)[:, : args.num_const * 3]
        dat = dat.reshape(dat.shape[0], -1, 3)
    else:
        assert False, "Filetype for bg not supported"
    dat = np.delete(dat, np.where(dat[:, 0, 0] == 0)[0], axis=0)
    dat[dat == -1] = 0

    file_val = file.replace("_train_", "_val_")

    if file.endswith("npz"):
        dat_val = np.load(file_val)["jets"][: args.num_events_val, : args.num_const]
    elif file_val.endswith("h5"):
        dat_val = pd.read_hdf(file_val, key="discretized", stop=args.num_events_val)
        dat_val = dat_val.to_numpy(dtype=np.int64)[:, : args.num_const * 3]
        dat_val = dat_val.reshape(dat_val.shape[0], -1, 3)
    else:
        assert False, "Filetype for bg not supported"
    dat_val = np.delete(dat_val, np.where(dat_val[:, 0, 0] == 0)[0], axis=0)
    dat_val[dat_val == -1] = 0

    return dat, dat_val


def get_dataloader(bgf, sigf):
    bg, bg_val = load_data(bgf)
    sig, sig_val = load_data(sigf)

    print(f"Using bg {bg.shape} from {bgf} and sig {sig.shape} from {sigf}")

    dat = np.concatenate((bg, sig), 0)
    lab = np.append(np.zeros(len(bg)), np.ones(len(sig)))
    padding_mask = dat[:, :, 0] != 0

    idx = np.random.permutation(len(dat))
    dat = torch.tensor(dat[idx])
    lab = torch.tensor(lab[idx])
    padding_mask = torch.tensor(padding_mask[idx])

    dat_val = np.concatenate((bg_val, sig_val), 0)
    lab_val = np.append(np.zeros(len(bg_val)), np.ones(len(sig_val)))
    padding_mask_val = dat_val[:, :, 0] != 0

    idx_val = np.random.permutation(len(dat_val))
    dat_val = torch.tensor(dat_val[idx_val])
    lab_val = torch.tensor(lab_val[idx_val])
    padding_mask_val = torch.tensor(padding_mask_val[idx_val])

    train_set = TensorDataset(
        dat,
        padding_mask,
        lab,
    )
    val_set = TensorDataset(
        dat_val,
        padding_mask_val,
        lab_val,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


def plot_rocs(model, val_loader, tag):
    labels = []
    preds = []
    model.eval()
    with torch.no_grad():
        for x, padding_mask, label in tqdm(val_loader, total=len(val_loader), desc=f"Validation {tag}"):
            x = x.to(device)
            padding_mask = padding_mask.to(device)
            label = label.to(device)

            logits = model(x, padding_mask)
            preds.append(logits.cpu().numpy())
            labels.append(label.cpu().numpy())

    preds = np.concatenate(preds, 0)
    labels = np.concatenate(labels, 0)
    fpr, tpr, _ = roc_curve(labels, preds)
    auc = roc_auc_score(labels, preds)
    print(f"AUC {tag}: {auc}")
    fig, ax = plt.subplots(constrained_layout=True)
    ax.plot(tpr, 1.0 / fpr, label=f"AUC {auc:.4f}")
    ax.set_yscale("log")
    ax.grid(which="both")
    ax.set_ylim(0.9, 3e3)
    ax.legend()
    fig.savefig(os.path.join(args.log_dir, f"roc_{tag}.png"))

    fig, ax = plt.subplots(constrained_layout=True)
    ax.hist(preds[labels == 0], histtype="step", density=True, bins=30, label="Bg")
    ax.hist(preds[labels == 1], histtype="step", density=True, bins=30, label="Sig")
    ax.legend()
    fig.savefig(os.path.join(args.log_dir, f"preds_{tag}.png"))

    np.savez(os.path.join(args.log_dir, f"preds_{tag}.npz"), preds=preds, labels=labels)
    plt.close(fig)


def orig_load_opt_dict(model_path_in, path_to_state_dict):
    checkpoint = torch.load(path_to_state_dict)

    state_dict = checkpoint['opt_state_dict_best']
    state_keys = list(state_dict['state'].keys())
    # Identify the last key
    last_keys = state_keys[-2:]
    # Remove the last entry
    for last_key in reversed(last_keys):
        del state_dict['state'][last_key]

        state_dict.get('param_groups')[0].get('params').pop(last_key)


    new_path = os.path.join(model_path_in, 'modified_opt_state_dict.pt')
    torch.save(state_dict, new_path)

    checkpoint_mod = torch.load(new_path)

    return checkpoint_mod


def GetLast2Layers(state_dict):
    state_keys = list(state_dict.get('param_groups')[0].get('params'))
    last_keys = state_keys[-2:]

    last2paramgroups = []
    last2state = []


    last2paramgroups.append(state_dict.get('param_groups')[0].get('params')[last_keys[0]])
    last2paramgroups.append(state_dict.get('param_groups')[0].get('params')[last_keys[1]])

    return last2paramgroups, last2state, last_keys


def AddLayersToDict(filtered_state_dict, last2state, last2paramgroups, last_keys):
    filtered_state_dict.get('param_groups')[0].get('params').append(last2paramgroups[0])
    filtered_state_dict.get('param_groups')[0].get('params').append(last2paramgroups[1])
    return filtered_state_dict


if __name__ == "__main__":
    args = parse_input()
    save_arguments(args)

    set_seeds(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}")

    num_features = 3
    num_bins = tuple(args.num_bins)

    print(f"Using bins: {num_bins}")

    train_loader, val_loader = get_dataloader(args.bg, args.sig)

    original_model = torch.load(os.path.join(args.model_path_in, args.model_name),)
    model = JetTransformerClassifierFine(original_model,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_features=num_features,
        dropout=args.dropout,
        num_const=args.num_const,
    )
    model.to(device)
    path_to_sate_dict = os.path.join(args.model_path_in, 'opt_state_dict_best.pt')

    filtered_opt_state_dict=orig_load_opt_dict(args.model_path_in,path_to_sate_dict)
    
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    last2paramgroups, last2state,last_keys=GetLast2Layers(opt.state_dict())
    filtered_opt_state_dict= AddLayersToDict(filtered_opt_state_dict,last2state,last2paramgroups,last_keys)
    opt.load_state_dict(filtered_opt_state_dict)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode='min',
        factor=args.learnrate_factor,
        patience=args.patience,
        verbose=True,
        min_lr=1e-8,
    )

    scaler = torch.cuda.amp.GradScaler()

    logger = SummaryWriter(args.log_dir)
    global_step = 0
    loss_list = []
    loss_list_epoch=[]
    val_list_epoch=[]

    min_val_loss = np.inf
    epochs_no_improve = 0

    for epoch in range(args.num_epochs):
        model.train()
        loss_list_here=[]
        for x, padding_mask, label in tqdm(
            train_loader, total=len(train_loader), desc=f"Training Epoch {epoch + 1}"
        ):
            opt.zero_grad()
            x = x.to(device)
            padding_mask = padding_mask.to(device)
            label = label.to(device)

            with torch.cuda.amp.autocast():
                logits = model(x, padding_mask)
                loss = model.loss(logits, label.view(-1, 1))

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            loss_list.append(loss.cpu().detach().numpy())
            loss_list_here.append(loss.cpu().detach().numpy())
            if (global_step + 1) % args.logging_steps == 0:
                logger.add_scalar("Train/Loss", np.mean(loss_list), global_step)
                logger.add_scalar("Train/LR", opt.param_groups[0]['lr'], global_step)
                loss_list = []
            global_step += 1

        model.eval()
        with torch.no_grad():
            val_loss = []
            for x, padding_mask, label in tqdm(
                val_loader, total=len(val_loader), desc=f"Validation Epoch {epoch + 1}"
            ):
                x = x.to(device)
                padding_mask = padding_mask.to(device)
                label = label.to(device)

                logits = model(x, padding_mask)
                loss = model.loss(logits, label.view(-1, 1))
                val_loss.append(loss.cpu().detach().numpy())

            val_loss = np.mean(val_loss)
            logger.add_scalar("Val/Loss", val_loss, global_step)

            # Scheduler mit Val Loss informieren
            scheduler.step(val_loss)

            if val_loss < min_val_loss:
                min_val_loss = val_loss
                epochs_no_improve = 0
                save_model(model, args.log_dir, "best")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= args.max_patience:

                    print(f"Stopping prematurely at epoch {epoch+1}")
                    break

        save_model(model, args.log_dir, "last")
        save_opt_states(
            optimizer=opt, scheduler=scheduler, scaler=scaler, log_dir=args.log_dir
        )

        loss_list_epoch.extend(loss_list_here)
        val_list_epoch.append(val_loss)

    model = load_model(os.path.join(args.log_dir, "model_best.pt"))
    plot_rocs(model, val_loader, tag="best")