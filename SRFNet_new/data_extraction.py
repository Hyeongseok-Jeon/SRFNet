import torch.nn as nn
from get_config import get_config
import argparse
from data import ArgoDataset
from torch.utils.data import DataLoader
import pickle5 as pickle
import torch
import os
import time
from tqdm import tqdm
import sys
import shutil
import numpy as np


def from_numpy(data):
    """Recursively transform numpy.ndarray to torch.Tensor.
    """
    if isinstance(data, dict):
        for key in data.keys():
            data[key] = from_numpy(data[key])
    if isinstance(data, list) or isinstance(data, tuple):
        data = [from_numpy(x) for x in data]
    if isinstance(data, np.ndarray):
        """Pytorch now has bool type."""
        data = torch.from_numpy(data)
    return data

def collate_fn(batch):
    batch = from_numpy(batch)
    return_batch = dict()
    # Batching by use a list for non-fixed size
    for key in batch[0].keys():
        return_batch[key] = [x[key] for x in batch]

    for keys in return_batch.keys():
        return_batch[keys] = [return_batch[keys][i][0] for i in range(len(return_batch[keys]))]

    orig_tot = torch.cat([return_batch['orig'][i].unsqueeze(dim=0) for i in range(len(return_batch['orig']))])  # (batch num, 2)
    batch_num = orig_tot.shape[0]
    gt_preds_tot = torch.cat(return_batch['gt_preds'])  # (vehicle num, 30, 2)
    has_preds_tot = torch.cat(return_batch['has_preds'])  # (vehicle num, 30)
    theta_tot = torch.cat([torch.from_numpy(np.asarray([[return_batch['theta'][i]]])) for i in range(len(return_batch['theta']))])  # (batch num, 1)
    rot_tot = torch.cat([return_batch['rot'][i].unsqueeze(dim=0) for i in range(len(return_batch['rot']))])  # (batch num, 2, 2)
    feats_tot = torch.cat(return_batch['feats'])  # (vehicle num, 20, 3)
    ego_feats_tot = torch.cat([return_batch['ego_feats'][i].unsqueeze(dim=0) for i in range(len(return_batch['ego_feats']))])  # (batch num, 50, 3)
    ctrs_tot = torch.cat(return_batch['ctrs'])  # (vehicle num, 2)
    action_input_tot = torch.cat([return_batch['action_input'][i].unsqueeze(dim=0) for i in range(len(return_batch['action_input']))])  # (batch num, 6, 30, 204)
    init_pred_global_reg_tot = torch.cat([return_batch['init_pred_global'][i]['reg'][0] for i in range(len(return_batch['init_pred_global']))], dim=0)  # (vehicle num, 6, 30, 2)
    init_pred_global_cls_tot = torch.cat([return_batch['init_pred_global'][i]['cls'][0] for i in range(len(return_batch['init_pred_global']))], dim=0)  # (vehicle num, 6)
    vehicle_per_batch = torch.cat([torch.tensor(return_batch['gt_preds'][i].shape[0]).unsqueeze(dim=0) for i in range(len(return_batch['gt_preds']))], dim=0)
    vehicle_per_batch_tmp = torch.cat((torch.tensor([0.], dtype=torch.float32, device=vehicle_per_batch.device), vehicle_per_batch))
    idx = []
    for i in range(batch_num + 1):
        idx.append(int(sum(vehicle_per_batch_tmp[j + 1] for j in range(i))))
    vehicle_num = gt_preds_tot.shape[0]
    data_num = 12
    max_vehicle_in_batch = torch.max(vehicle_per_batch)

    mask = torch.zeros(size=[batch_num, data_num, max_vehicle_in_batch, 50, 30, 2])
    mask[:, 0, 0, :2, 0, 0] = orig_tot
    mask[:, 3, 0, :1, 0, 0] = theta_tot
    mask[:, 4, 0, :2, :2, 0] = rot_tot
    mask[:, 6, 0, :50, :3, 0] = ego_feats_tot
    mask[:, 11, 0, 0, 0, 0] = vehicle_per_batch

    for i in range(batch_num):
        mask[i, 1, :vehicle_per_batch[i], :30, :2, 0] = gt_preds_tot[idx[i]:idx[i + 1], :, :]
        mask[i, 2, :vehicle_per_batch[i], :30, 0, 0] = has_preds_tot[idx[i]:idx[i + 1], :]
        mask[i, 5, :vehicle_per_batch[i], :20, :3, 0] = feats_tot[idx[i]:idx[i + 1], :, :]
        mask[i, 7, :vehicle_per_batch[i], :2, 0, 0] = ctrs_tot[idx[i]:idx[i + 1], :]
        mask[i, 9, :vehicle_per_batch[i], :6, :30, :2] = init_pred_global_reg_tot[idx[i]:idx[i + 1], :, :, :]
        mask[i, 10, :vehicle_per_batch[i], :6, 0, 0] = init_pred_global_cls_tot[idx[i]:idx[i + 1], :]

    return mask, action_input_tot, return_batch['graph'], return_batch['file_name']

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
cur_dir = os.getcwd()
dataset = ArgoDataset(config["train_split"], config, train=False)
train_loader = DataLoader(
    dataset,
    batch_size=1,
    num_workers=config["workers"],
    collate_fn=collate_fn,
    drop_last=True,
)
os.makedirs(cur_dir + '/SRFNet_new/dataset/preprocess_GAN/val', exist_ok=True)
for i, data in tqdm(enumerate(train_loader)):
    file_name = data[3][0].name[:-4]
    if not(os.path.isfile(cur_dir + '/SRFNet_new/dataset/preprocess_GAN/val/'+ file_name + '.pickle')):
        with open(cur_dir + '/SRFNet_new/dataset/preprocess_GAN/val/'+ file_name + '.pickle', 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
