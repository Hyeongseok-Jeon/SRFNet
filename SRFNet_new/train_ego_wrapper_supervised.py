import torch.nn as nn
import model_ego_wrapper_supervised as model
from model_ego_wrapper_supervised import Loss, PostProcess
from get_config import get_config
import argparse
from data import ArgoDataset, collate_fn
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

base_net, weight, opt = lanegcn.get_model(config)
root_path = os.path.join(os.path.abspath(os.curdir))
pre_trained_weight = torch.load(os.path.join(root_path, "LaneGCN/pre_trained") + '/36.000.ckpt')
pretrained_dict = pre_trained_weight['state_dict']
base_net.load_state_dict(pretrained_dict)

net = model.model_class(config, args, base_net)
model = nn.DataParallel(net)
model.cuda()

dataset = ArgoDataset(config["train_split"], config, train=False)
train_loader = DataLoader(
    dataset,
    batch_size=config["batch_size"],
    shuffle=True,
    num_workers=config["workers"],
    collate_fn=collate_fn,
    drop_last=True,
)

# Data loader for evaluation
# dataset = ArgoDataset(config["val_split"], config, train=False)
# val_loader = DataLoader(
#     dataset,
#     batch_size=config["val_batch_size"],
#     num_workers=config["val_workers"],
#     shuffle=True,
#     collate_fn=collate_fn,
# )
l1loss = nn.SmoothL1Loss()
loss_logging = Loss(config)
post_process = PostProcess(config)
os.makedirs(config['save_dir'], exist_ok=True)

log = os.path.join(config['save_dir'], "log")

if not os.path.exists(config['save_dir']):
    os.makedirs(config['save_dir'])
sys.stdout = Logger(log)

src_dirs = [root_path]
dst_dirs = [os.path.join(config['save_dir'], "files")]
for src_dir, dst_dir in zip(src_dirs, dst_dirs):
    files = [f for f in os.listdir(src_dir) if f.endswith(".py")]
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    for f in files:
        shutil.copy(os.path.join(src_dir, f), os.path.join(dst_dir, f))

start_time = time.time()
for epoch in range(config["num_epochs"]):
    metrics = dict()
    for i, data in tqdm(enumerate(train_loader)):
        with torch.no_grad():
            actors, actors_idcs = base_net(data)
        outputs = model(data[0], data[1], actors, actors_idcs)
        batch_num = data[0].shape[0]
        vehicle_per_batch = data[0][:, 11, 0, 0, 0, 0]
        gt_preds = [data[0][i, 1, :int(vehicle_per_batch[i]), :30, :2, 0] for i in range(batch_num)]
        gt = torch.cat([torch.repeat_interleave(gt_preds[i][1:2, :, :], 6, dim=0).unsqueeze(dim=0) for i in range(len(gt_preds))], dim=0)
        loss = l1loss(outputs[:,1,:,:,:].cpu(), gt)

        # actors should be tensor
        output_reform = dict()
        cls = [outputs[i:i+1, 0, :, 0, 0] for i in range(outputs.shape[0])]
        reg = [outputs[i:i+1, 1, :, :, :] for i in range(outputs.shape[0])]
        output_reform['cls'] = cls
        output_reform['reg'] = reg
        output_reform = [output_reform]

        if i == 0:
            loss_out = loss_logging(output_reform[0], data)
            post_out = post_process(output_reform[0], data)
            post_process.append(metrics, loss_out, post_out)
            dt = time.time() - start_time
            post_process.display(metrics, dt, epoch, 0.001)

        opt.zero_grad()
        loss.backward()
        opt.step(epoch)

        loss_out = loss_logging(output_reform[0], data)
        post_out = post_process(output_reform[0], data)
        post_process.append(metrics, loss_out, post_out)

    save_ckpt(net, opt, config['save_dir'], epoch)
    dt = time.time() - start_time
    post_process.display(metrics, dt, epoch, 0.001)
    start_time = time.time()
    metrics = dict()

    if epoch % 2 == 1:
        metrics = dict()
        for i, data in tqdm(enumerate(val_loader)):
            with torch.no_grad():
                actors = base_net(data)
                outputs = model(data[0], data[1], actors)
            # actors should be tensor

                batch_num = data[0].shape[0]
                vehicle_per_batch = data[0][:, 11, 0, 0, 0, 0]
                gt_preds = [data[0][i, 1, :int(vehicle_per_batch[i]), :30, :2, 0] for i in range(batch_num)]

                pred = torch.cat([torch.cat(outputs[0]['reg'], dim=0)[i, :, :, :] for i in range(len(outputs[0]['reg']))], dim=0).cpu()
                gt = torch.cat([torch.repeat_interleave(gt_preds[i][1:2, :, :], 6, dim=0) for i in range(len(gt_preds))], dim=0)
                loss = l1loss(pred, gt)

                loss_out = loss_logging(outputs[0], data)
                post_out = post_process(outputs[0], data)
                post_process.append(metrics, loss_out, post_out)
        dt = time.time() - start_time
        post_process.display(metrics, dt, epoch, 0.001)
        start_time = time.time()
        metrics = dict()