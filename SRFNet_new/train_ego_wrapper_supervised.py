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

net = model.model(config, args, base_net)
model = nn.DataParallel(net)
model.cuda()

dataset = ArgoDataset(config["train_split"], config, train=False)
train_loader = DataLoader(
    dataset,
    batch_size=config["batch_size"],
    shuffle=True,
    num_workers=config["workers"],
    collate_fn=collate_fn,
    pin_memory=True,
    drop_last=True,
)

# Data loader for evaluation
dataset = ArgoDataset(config["val_split"], config, train=False)
val_loader = DataLoader(
    dataset,
    batch_size=config["val_batch_size"],
    num_workers=config["val_workers"],
    shuffle=True,
    collate_fn=collate_fn,
    pin_memory=True,
)
l1loss = nn.SmoothL1Loss()
loss_logging = Loss(config)
post_process = PostProcess(config)

for epoch in range(config["num_epochs"]):
    metrics = dict()
    for i, data in enumerate(train_loader):
        print(i)
        with torch.no_grad():
            actors = base_net(data)
        outputs = model(data, actors)
        pred = torch.cat([torch.cat(outputs[0]['reg'], dim=0)[i, :, :, :] for i in range(len(outputs[0]['reg']))], dim=0).cpu()
        gt = torch.cat([torch.repeat_interleave(data['gt_preds'][i][1:2,:,:], 6, dim=0) for i in range(len(data['gt_preds']))], dim=0)
        loss = l1loss(pred, gt)

        opt.zero_grad()
        loss.backward()
        opt.step(epoch)

        loss_out = loss_logging(outputs[0], data)
        post_out = post_process(outputs[0], data)
        post_process.append(metrics, loss_out, post_out)

    save_ckpt(net, opt, config['save_dir'], epoch)

    start_time = time.time()
    dt = time.time() - start_time
    post_process.display(metrics, dt, epoch, 0.001)
    start_time = time.time()

