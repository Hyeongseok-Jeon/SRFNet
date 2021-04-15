import torch.nn as nn
import SRFNet_new.model_ego_wrapper_supervised as model
from SRFNet_new.get_config import get_config
import argparse
from SRFNet_new.data import ArgoDataset, collate_fn
from torch.utils.data import DataLoader
from SRFNet_new.baselines.LaneGCN import lanegcn
import torch
import os

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
    "--case", default="vanilla_gan", type=str
)
parser.add_argument(
    "--gpu_id", default=0, type=int
)
parser.add_argument("--mode", default='client')
parser.add_argument("--port", default=52162)
args = parser.parse_args()
config = get_config()

base_net, weight = lanegcn.get_model(config)
root_path = os.path.join(os.path.abspath(os.curdir))
pre_trained_weight = torch.load(os.path.join(root_path, "LaneGCN/pre_trained") + '/36.000.ckpt')
pretrained_dict = pre_trained_weight['state_dict']
base_net.load_state_dict(pretrained_dict)

net = model.model(config, args, base_net)
model = nn.DataParallel(net)
model.cuda()

dataset = ArgoDataset(config["train_split"], config, train=True)
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

for i, (inputs, labels) in enumerate(train_loader):
    outputs = model(inputs)
    loss = criterion(outputs, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()