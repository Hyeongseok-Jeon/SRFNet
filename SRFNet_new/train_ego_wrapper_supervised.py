import torch.nn as nn
import model_ego_wrapper_supervised as model
from model_ego_wrapper_supervised import Loss, PostProcess
from get_config import get_config
import argparse
from SRF_data_loader import SRF_data_loader, collate_fn
from torch.utils.data import DataLoader
from baselines.LaneGCN import lanegcn
import torch
import os
import time
from tqdm import tqdm
import sys
from utils import Logger, load_pretrain
import shutil


def save_ckpt(net, opt, save_dir, epoch):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    state_dict = net.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].cpu()

    save_name = "%3.3f.ckpt" % epoch
    torch.save(
        {"epoch": epoch, "state_dict": state_dict, "opt_state": opt.opt.state_dict()},
        os.path.join(save_dir, save_name),
    )


parser = argparse.ArgumentParser(description="Fuse Detection in Pytorch")
parser.add_argument(
    "--base_line", default="LaneGCN", type=str, metavar="MODEL", help="model name"
)
parser.add_argument("--eval", action="store_true")
parser.add_argument(
    "--resume", default="", type=str, metavar="RESUME", help="checkpoint path"
)
parser.add_argument(
    "--weight", default="", type=str, metavar="WEIGHT", help="checkpoint path"
)
parser.add_argument(
    "--case", default="supervised wrapper", type=str
)
parser.add_argument(
    "--gpu_id", default=0, type=int
)
parser.add_argument("--mode", default='client')
parser.add_argument("--port", default=52162)
args = parser.parse_args()
config = get_config(args)
config['gpu_id'] = args.gpu_id

time.sleep(10)
