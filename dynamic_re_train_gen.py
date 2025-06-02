import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import time, os
from argparse import ArgumentParser

from torch.utils.tensorboard import SummaryWriter

from model import JetTransformer
from helpers_train import *

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
torch.multiprocessing.set_sharing_strategy("file_system")

def set_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def save_arguments(args):
    tmp = args.log_dir + "_" + args.name_sufix
    args.log_dir = tmp
    os.makedirs(args.log_dir, exist_ok=True)
    with open(os.path.join(args.log_dir, "arguments.txt"), "w") as f:
        for k, v in vars(args).items():
            f.write(f"{k:20s} {v}\n")
    return args

parser = ArgumentParser()
parser.add_argument("--model_path_in", type=str, default="models/test", help="folder of model to continue training e.g. /models/TTBar_10m_20e")
parser.add_argument("--model_name", type=str, default="model_best.pt")
parser.add_argument("--model_path", type=str, default="models/test_2")
parser.add_argument("--batch_size", type=int, default=100)
parser.add_argument("--num_const", type=int, default=128)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--lr", type=float, default=.001)
parser.add_argument("--weight_decay", type=float, default=.00001)
parser.add_argument("--num_epochs", type=int, default=50, help="max number of epochs")
parser.add_argument("--output", type=str, default='linear')
parser.add_argument("--name_sufix", type=str, default="A1B2C3D")
parser.add_argument("--data_path", type=str)
parser.add_argument("--num_events", type=int, default=10000)
parser.add_argument("--num_events_val", type=int, default=500000)
parser.add_argument("--num_bins", type=int, nargs=3, default=[41, 31, 31])
parser.add_argument("--reverse", action="store_true")
parser.add_argument("--start_token", action="store_true")
parser.add_argument("--end_token", action="store_true")
parser.add_argument("--limit_const", action="store_true")
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--log_dir", type=str, default="models/test")
parser.add_argument("--checkpoint_steps", type=int, default=100000)
parser.add_argument("--logging_steps", type=int, default=10)
parser.add_argument("--contin", action="store_true", help="continue training exactly where it stopped: same optimizer, etc.")
parser.add_argument("--global_step", type=int, default=0)
parser.add_argument("--learnrate_factor", type=float, default=0.5, help="after <patience> epochs without improvement, multiply learnrate by <learnrate_factor>")
parser.add_argument("--patience", type=int, default=2, help="after <patience> epochs without improvement, multiply learnrate by <learnrate_factor>")
parser.add_argument("--max_patience", type=int, default=5, help="stop training prematurely after <max_patience> epochs without improvement")

args = parser.parse_args()

set_seeds(args.seed)
device = "cuda" if torch.cuda.is_available() else "cpu"
assert device == "cuda", "Not running on GPU"
num_features = 3
num_bins = tuple(args.num_bins)

train_loader = load_data(
    path=args.data_path,
    n_events=args.num_events,
    num_features=num_features,
    num_bins=num_bins,
    num_const=args.num_const,
    reverse=args.reverse,
    start_token=args.start_token,
    end_token=args.end_token,
    limit_const=args.limit_const,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
)

val_loader = load_data(
    path=args.data_path.replace("train", "val"),
    n_events=args.num_events_val,
    num_features=num_features,
    num_bins=num_bins,
    num_const=args.num_const,
    reverse=args.reverse,
    start_token=args.start_token,
    end_token=args.end_token,
    limit_const=args.limit_const,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
)

args = save_arguments(args)

model = torch.load(os.path.join(args.model_path_in, args.model_name))
model.classifier = False
model.to(device)

opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=args.learnrate_factor, patience=args.patience, verbose=True)
scaler = torch.cuda.amp.GradScaler()

if args.contin:
    load_opt_states_best(opt, scheduler, scaler, args.model_path_in)
    print("Loaded optimizer")

logger = SummaryWriter(args.log_dir)
global_step = args.global_step
loss_list, perplexity_list = [], []

best_val_loss = float('inf')
no_improve_epochs = 0
max_patience = args.max_patience

print("training...")
for epoch in range(args.num_epochs):
    model.train()
    for x, padding_mask, true_bin in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
        x, padding_mask, true_bin = x.to(device), padding_mask.to(device), true_bin.to(device)
        opt.zero_grad()

        with torch.cuda.amp.autocast():
            logits = model(x, padding_mask)
            loss = model.loss(logits, true_bin)
            with torch.no_grad():
                perplexity = model.probability(logits, padding_mask, true_bin, perplexity=True)

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        loss_list.append(loss.item())
        perplexity_list.append(perplexity.mean().item())

        if (global_step + 1) % args.logging_steps == 0:
            logger.add_scalar("Train/Loss", np.mean(loss_list), global_step)
            logger.add_scalar("Train/Perplexity", np.mean(perplexity_list), global_step)
            logger.add_scalar("Train/LR", opt.param_groups[0]['lr'], global_step)
            loss_list, perplexity_list = [], []

        if args.checkpoint_steps and (global_step + 1) % args.checkpoint_steps == 0:
            save_model(model, args.log_dir, f"checkpoint_{global_step+1}")

        global_step += 1

    model.eval()
    val_loss, val_perplexity = [], []
    with torch.no_grad():
        for x, padding_mask, true_bin in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}"):
            x, padding_mask, true_bin = x.to(device), padding_mask.to(device), true_bin.to(device)
            logits = model(x, padding_mask)
            loss = model.loss(logits, true_bin)
            perplexity = model.probability(logits, padding_mask, true_bin, perplexity=True)
            val_loss.append(loss.item())
            val_perplexity.append(perplexity.mean().item())

    val_loss_mean = np.mean(val_loss)
    logger.add_scalar("Val/Loss", val_loss_mean, global_step)
    logger.add_scalar("Val/Perplexity", np.mean(val_perplexity), global_step)
    scheduler.step(val_loss_mean)

    if val_loss_mean < best_val_loss:
        best_val_loss = val_loss_mean
        no_improve_epochs = 0
        save_model(model, args.log_dir, "best")
        save_opt_states_best(opt, scheduler, scaler, args.log_dir)
        print(f"New best model saved with val loss {val_loss_mean:.4f}")
    else:
        no_improve_epochs += 1

    save_model(model, args.log_dir, "last")
    save_opt_states(opt, scheduler, scaler, args.log_dir)

    if no_improve_epochs >= max_patience:
        print(f"Early stopping after {epoch+1} epochs with no improvement.")
        break
