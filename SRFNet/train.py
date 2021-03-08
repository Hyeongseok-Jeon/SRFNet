import argparse
from torch.utils.data import DataLoader

from torch.autograd import Variable
import torch

import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import itertools
import numpy as np
import sys
import pickle
import warnings
import time
import datetime
import os
import json
import argparse
import numpy as np
import random
import sys
sys.path.extend(['/home/jhs/Desktop/SRFNet/LaneGCN'])
import time
import shutil
from importlib import import_module
from numbers import Number
from tqdm import tqdm
import torch
from torch.utils.data import Sampler, DataLoader
import horovod.torch as hvd
from torch.utils.data.distributed import DistributedSampler
from mpi4py import MPI
from SRFNet.data import ArgoDataset as Dataset, collate_fn
from LaneGCN.lanegcn import PostProcess
from SRFNet.config import get_config
from LaneGCN.utils import Optimizer
from SRFNet.model import Net, Loss
warnings.filterwarnings("ignore")

root_path = os.path.join(os.path.abspath(os.curdir))
sys.path.insert(0, root_path)

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument("--mode", default='client')
parser.add_argument("--port", default=52162)
args = parser.parse_args()

def main():
    config = get_config(root_path)
    # post processing function
    post_process = PostProcess(config)

    # data loader for training
    dataset = Dataset(config["train_split"], config, train=True)
    train_loader = DataLoader(dataset,
                              batch_size=config["batch_size"],
                              num_workers=config["workers"],
                              collate_fn=collate_fn,
                              shuffle=True,
                              pin_memory=True,
                              drop_last=True)

    # Data loader for evaluation
    dataset = Dataset(config["val_split"], config, train=False)
    val_loader = DataLoader(dataset,
                            batch_size=config["val_batch_size"],
                            num_workers=config["val_workers"],
                            collate_fn=collate_fn,
                            shuffle=True,
                            pin_memory=True)

    net = Net(config)
    pre_trained_weight = torch.load(os.path.join(root_path, "LaneGCN/pre_trained") + '/36.000.ckpt')
    pretrained_dict = pre_trained_weight['state_dict']
    new_model_dict = net.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in new_model_dict}
    new_model_dict.update(pretrained_dict)
    net.load_state_dict(new_model_dict)
    net = net.cuda()

    opt = Optimizer(net.parameters(), config)
    loss = Loss(config)

    train(config, train_loader, net, loss, post_process, opt, val_loader)


def train(config, train_loader, net, loss, post_process, opt, val_loader=None):
    net.train()

    save_iters = config["save_freq"]
    display_iters = config["display_iters"]
    val_iters = config["val_iters"]

    start_time = time.time()
    metrics = dict()
    for epoch in range(config['num_epochs']):
        for i, data in enumerate(train_loader):
            print(i)
            data = dict(data)
            output = net(data)
            loss_out = loss(output, data)
            post_out = post_process(output, data)
            post_process.append(metrics, loss_out, post_out)

            opt.zero_grad()
            loss_out["loss"].backward()
            lr = opt.step(epoch)


        if epoch % save_iters == save_iters - 1:
            save_ckpt(net, opt, config["save_dir"], epoch)

        if epoch % display_iters == display_iters - 1:
            dt = time.time() - start_time
            post_process.display(metrics, dt, epoch, lr)
            start_time = time.time()
            metrics = dict()

        if epoch % val_iters == val_iters - 1:
            val(config, val_loader, net, loss, post_process, epoch)

def val(config, data_loader, net, loss, post_process, epoch):
    net.eval()
    start_time = time.time()
    metrics = dict()
    for i, data in enumerate(data_loader):
        data = dict(data)
        with torch.no_grad():
            output = net(data)
            loss_out = loss(output, data)
            post_out = post_process(output, data)
            post_process.append(metrics, loss_out, post_out)

    dt = time.time() - start_time
    post_process.display(metrics, dt, epoch)
    net.train()


def save_ckpt(net, opt, save_dir, epoch):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    state_dict = net.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].cpu()

    save_name = "%3.3f.ckpt" % epoch
    torch.save(
        {"epoch": epoch, "state_dict": state_dict, "opt_state": opt.opt.state_dict()},
        os.path.join(save_dir, save_name),
    )
