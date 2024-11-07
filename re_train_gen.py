import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import time, os
from argparse import ArgumentParser

import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from model import JetTransformer

from tqdm import tqdm
from helpers_train import *

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

torch.multiprocessing.set_sharing_strategy("file_system")


def set_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def save_arguments(args):
    '''
    tmp = args.log_dir
    i = 0
    while os.path.isdir(tmp):
        i += 1
        #tmp = args.log_dir + f"_{i}"
        tmp = args.log_dir+"_"+ args.name_sufix
    '''
    tmp = args.log_dir+"_"+ args.name_sufix
    args.log_dir = tmp
    os.makedirs(args.log_dir)
    with open(os.path.join(args.log_dir, "arguments.txt"), "w") as f:
        arg_dict = vars(args)
        for k, v in arg_dict.items():
            f.write(f"{k:20s} {v}\n")
    return args

parser = ArgumentParser()
parser.add_argument("--model_path_in", type=str, default="models/test")
parser.add_argument("--model_name", type=str, default="model_best.pt")
parser.add_argument("--model_path", type=str, default="models/test_2")

#parser.add_argument("--num_samples", type=int, default=1000)
parser.add_argument("--batch_size", type=int, default=100)
parser.add_argument("--num_const", type=int, default=50)
parser.add_argument("--seed", type=int, default=0)
#parser.add_argument("--trunc", type=float, default=None)
parser.add_argument("--lr", type=float, default=.001)

parser.add_argument("--weight_decay", type=float, default=.00001)
parser.add_argument("--num_epochs", type=int, default=30)
parser.add_argument("--output", type=str, default='linear')


parser.add_argument(
        "--name_sufix", type=str, default="A1B2C3D", help="name of train dir"
    )

parser.add_argument(
        "--data_path",
        type=str,
        default="/hpcwork/bn227573/top_benchmark/train_qcd_30_bins.h5",
        help="Path to training data file",
    )



#parser.add_argument( "--model_path", type=str,default=None help="Model file to load")
parser.add_argument(
        "--num_events", type=int, default=10000, help="Number of events for training"
    )
    
parser.add_argument(
        "--num_events_val", type=int, default=500000, help="Number of events for training"
    )

parser.add_argument(
        "--num_bins",
        type=int,
        nargs=3,
        default=[41, 31, 31],
        help="Number of bins per feature",
    )
parser.add_argument(
        "--reverse", action="store_true", help="Whether to reverse pt order"
    )
parser.add_argument(
        "--start_token",
        action="store_true",
        help="Whether to use a start particle (learn first particle as well)",
    )
parser.add_argument(
        "--end_token",
        action="store_true",
        help="Whether to use a end particle (learn jet length as well)",
    )

parser.add_argument(
        "--limit_const",
        action="store_true",
        help="Only use jets with at least num_const constituents",
    )


parser.add_argument("--num_workers", type=int, default=4, help="Number of workers")

parser.add_argument(
        "--log_dir", type=str, default="models/test", help="Model directory"
    )

parser.add_argument(
        "--checkpoint_steps", type=int, default=100000, help="Model directory"
    )
parser.add_argument(
        "--logging_steps", type=int, default=10, help="Training steps between logging"
    )
parser.add_argument(
        "--contin",      action="store_true",
        help="continue training",
    )
parser.add_argument(
        "--global_step", type=int, default=0, help="Starting point of step counter"
    )

args = parser.parse_args()

set_seeds(args.seed)

device = "cuda" if torch.cuda.is_available() else "cpu"
assert device == "cuda", "Not running on GPU"
print('num_epochs')
print(args.num_epochs)
print('limit_const')
print(args.limit_const)
num_features = 3
num_bins = tuple(args.num_bins)
print('model path in')
print(args.model_path_in)

print("Loading training set")
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


print("Loading validation set")
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

#n_batches = args.num_samples // args.batchsize
#rest = args.num_samples % args.batchsize
args=save_arguments(args)
# Load model for sampling
print('model path in')
print(args.model_path_in)
print(os.path.join(args.model_path_in, args.model_name))
model = torch.load(os.path.join(args.model_path_in, args.model_name))
model.classifier = False
model.to(device)


# construct optimizer and auto-caster
opt = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
print(opt)
print(len(train_loader))
scheduler = get_cos_scheduler(int(args.num_epochs), len(train_loader), opt)
scaler = torch.cuda.amp.GradScaler()

if args.contin:
    load_opt_states_best(opt, scheduler, scaler, args.model_path_in)
    print("Loaded optimizer")

logger = SummaryWriter(args.log_dir)
global_step = args.global_step
loss_list = []
perplexity_list = []
mean_val_loss=9999999
print('training...')
for epoch in range(args.num_epochs):
    model.train()
    print('epoch'+str(epoch))
    for x, padding_mask, true_bin in tqdm(
        train_loader, total=len(train_loader), desc=f"Training Epoch {epoch + 1}"
    ):
        opt.zero_grad()
        x = x.to(device)
        padding_mask = padding_mask.to(device)
        true_bin = true_bin.to(device)

        with torch.cuda.amp.autocast():
            logits = model(x, padding_mask)
            loss = model.loss(logits, true_bin)
            with torch.no_grad():
                perplexity = model.probability(
                    logits,
                    padding_mask,
                    true_bin,
                    perplexity=True,
                    logarithmic=False,
                    topk=False
                )

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        scheduler.step()

        loss_list.append(loss.cpu().detach().numpy())
        perplexity_list.append(perplexity.mean().cpu().detach().numpy())

        if (global_step + 1) % args.logging_steps == 0:
            logger.add_scalar("Train/Loss", np.mean(loss_list), global_step)
            logger.add_scalar(
                "Train/Perplexity", np.mean(perplexity_list), global_step
            )
            logger.add_scalar("Train/LR", scheduler.get_last_lr()[0], global_step)
            loss_list = []
            perplexity_list = []

        if (args.checkpoint_steps != 0) and (
            (global_step + 1) % args.checkpoint_steps == 0
        ):
            save_model(model, args.log_dir, f"checkpoint_{global_step + 1}")

        global_step += 1

    model.eval()
    with torch.no_grad():
        val_loss = []
        val_perplexity = []
        for x, padding_mask, true_bin in tqdm(
            val_loader, total=len(val_loader), desc=f"Validation Epoch {epoch + 1}"
        ):
            x = x.to(device)
            padding_mask = padding_mask.to(device)
            true_bin = true_bin.to(device)

            logits = model(
                x,
                padding_mask,
            )
            loss = model.loss(logits, true_bin)
            perplexity = model.probability(
                logits, padding_mask, true_bin, perplexity=True, logarithmic=False
            )
            val_loss.append(loss.cpu().detach().numpy())
            val_perplexity.append(perplexity.mean().cpu().detach().numpy())

        logger.add_scalar("Val/Loss", np.mean(val_loss), global_step)
        logger.add_scalar("Val/Perplexity", np.mean(val_perplexity), global_step)
    
    if np.mean(val_loss) < mean_val_loss:
            print('new val loss:'+str(np.mean(val_loss))+'<'+str(mean_val_loss)+' saving new model as best' )
            save_model(model, args.log_dir, "best")
            mean_val_loss=np.mean(val_loss)
            save_opt_states_best(opt, scheduler, scaler, args.log_dir)
        
    save_model(model, args.log_dir, "last")
    save_opt_states(opt, scheduler, scaler, args.log_dir)
