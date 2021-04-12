# Copyright (c) 2020 Uber Technologies, Inc.
# Please check LICENSE for more detail


"""
Preprocess the data(csv), build graph from the HDMAP and saved as pkl
"""

import argparse
import os
import sys
sys.path.extend(['/home/jhs/Desktop/SRFNet'])
sys.path.extend(['/home/jhs/Desktop/SRFNet/LaneGCN'])
sys.path.extend(['/home/user/Desktop/SRFNet'])
sys.path.extend(['/home/user/Desktop/SRFNet/LaneGCN'])
sys.path.extend(['/home/user/data/HyeongseokJeon/infogan_pred/SRFNet'])
sys.path.extend(['/home/user/data/HyeongseokJeon/infogan_pred/SRFNet/LaneGCN'])

import pickle5 as pickle
import random

import time
from importlib import import_module

from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from SRFNet.code_by_jhs.data import ArgoDataset as Dataset, from_numpy, ref_copy, collate_fn
from SRFNet.model_lanegcn import get_model
import warnings
warnings.filterwarnings("ignore")

os.umask(0)

root_path = os.path.join(os.path.abspath(os.curdir))
sys.path.insert(0, root_path)

parser = argparse.ArgumentParser(
    description="Data preprocess for argo forcasting dataset"
)
parser.add_argument(
    "-m", "--model", default="lanegcn", type=str, metavar="MODEL", help="model name"
)
parser.add_argument(
    "-case", default="lanegcn", type=str, metavar="MODEL", help="model name"
)
parser.add_argument("--mode", default='client')
parser.add_argument("--port", default=52162)


def main():
    # Import all settings for experiment.
    args = parser.parse_args()
    config = dict()

    config["preprocess"] = True  # whether use preprocess or not
    config["preprocess_train"] = os.path.join(
        root_path,"SRFNet", "dataset", "preprocess", "train_crs_dist6_angle90.p"
    )
    config["preprocess_val"] = os.path.join(
        root_path, "SRFNet","dataset", "preprocess", "val_crs_dist6_angle90.p"
    )
    config['preprocess_test'] = os.path.join(root_path,"SRFNet", "dataset", 'preprocess', 'test_test.p')
    config["preprocess"] = True  # we use raw data to generate preprocess data
    config["val_workers"] = 32
    config["workers"] = 32
    config['cross_dist'] = 6
    config['cross_angle'] = 0.5 * np.pi
    config["train_split"] = os.path.join(
        root_path, "LaneGCN","dataset/train/data"
    )
    config["val_split"] = os.path.join(root_path, "LaneGCN", "dataset/val/data")
    config["test_split"] = os.path.join(root_path,"LaneGCN", "dataset/test_obs/data")
    config["batch_size"] = 32
    config["val_batch_size"] = 32
    config["rot_aug"] = False
    config["pred_range"] = [-100.0, 100.0, -100.0, 100.0]
    config["num_scales"] = 6

    config["pred_range"] = [-100.0, 100.0, -100.0, 100.0]
    config["num_scales"] = 6
    config["n_actor"] = 128
    config["n_map"] = 128
    config["actor2map_dist"] = 7.0
    config["map2actor_dist"] = 6.0
    config["actor2actor_dist"] = 100.0
    config["pred_size"] = 30
    config["pred_step"] = 1
    config["num_preds"] = config["pred_size"] // config["pred_step"]
    config["num_mods"] = 6
    config["cls_coef"] = 1.0
    config["reg_coef"] = 1.0
    config["mgn"] = 0.2
    config["cls_th"] = 2.0
    config["cls_ignore"] = 0.2
    config["GAT_dropout"] = 0.5
    config["GAT_Leakyrelu_alpha"] = 0.2
    config["GAT_num_head"] = config["n_actor"]
    config["SRF_conv_num"] = 4
    config["inter_dist_thres"] = 10
    config['gan_noise_dim'] = 128

    _, _, _, pre_model, _, _, _, _ = get_model(args)
    pre_trained_weight = torch.load(os.path.join(root_path, "LaneGCN/pre_trained") + '/36.000.ckpt')
    pretrained_dict = pre_trained_weight['state_dict']
    new_model_dict = pre_model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in new_model_dict}
    new_model_dict.update(pretrained_dict)
    pre_model.load_state_dict(new_model_dict)
    os.makedirs(os.path.join(root_path, 'SRFNet', 'dataset', 'preprocess_GAN'),exist_ok=True)

    gen('val', pre_model, config)
    # gen('test', pre_model, config)
    gen('train', pre_model, config)

def gen(mod, pre_model, config):
    if mod == 'train':
        train = True
        split = config["train_split"]
        data_num = 205942
        dir = config["preprocess_train"]
    elif mod == 'val':
        train = False
        split = config["val_split"]
        data_num = 39472
        dir = config["preprocess_val"]
    elif mod == 'test':
        train = False
        split = config["test_split"]
        data_num = 78143
        dir = config["preprocess_test"]

    # Data loader for training set
    dataset = Dataset(split, config, train=train)
    data_loader = DataLoader(
        dataset,
        batch_size=2,
        num_workers=2,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False,
    )

    t = time.time()
    for i, data in enumerate(tqdm(data_loader)):
        data = dict(data)
        with torch.no_grad():
            init_pred_global = pre_model(data)
        for j in range(len(data["idx"])):
            init_pred_global_con = init_pred_global[0].copy()
            init_pred_global_con['reg'][j] = init_pred_global_con['reg'][j][1:2, :, :, :].cpu()

            ego_fut_traj = to_numpy(data['gt_preds'][j][0:1, :, :])

            cl_cands = data['cl_cands']
            cl_cands_target = to_numpy(cl_cands[j][1])
            hid = reform(ego_fut_traj, cl_cands_target, init_pred_global_con['reg'][j].cpu())

            hid_np = hid.numpy()
            dict_batch = dict()
            dict_batch['cls'] = [init_pred_global[0]['cls'][j].cpu().numpy()]
            dict_batch['reg'] = [init_pred_global[0]['reg'][j].cpu().numpy()]
            init_pred_global_np = [dict_batch]

            init_pred_global_con_np = dict()
            init_pred_global_con_np['cls'] = [init_pred_global_con['cls'][j].cpu().numpy()]
            init_pred_global_con_np['reg'] = [init_pred_global_con['reg'][j].cpu().numpy()]

            dataset.split[data["idx"][j]]['data'] = hid_np
            dataset.split[data["idx"][j]]['init_pred_global'] = init_pred_global_np
            dataset.split[data["idx"][j]]['init_pred_global_con'] = init_pred_global_con_np


        if (i + 1) % 100 == 0:
            print(i, time.time() - t)
            t = time.time()

    dataset = PreprocessDataset(dataset.split, config, train=True)
    data_loader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        num_workers=config['workers'],
        shuffle=False,
        collate_fn=from_numpy,
        pin_memory=True,
        drop_last=False)

    modify(config, data_loader, dir)


def to_numpy(data):
    """Recursively transform torch.Tensor to numpy.ndarray.
    """
    if isinstance(data, dict):
        for key in data.keys():
            data[key] = to_numpy(data[key])
    if isinstance(data, list) or isinstance(data, tuple):
        data = [to_numpy(x) for x in data]
    if torch.is_tensor(data):
        data = data.numpy()
    return data


def to_int16(data):
    if isinstance(data, dict):
        for key in data.keys():
            data[key] = to_int16(data[key])
    if isinstance(data, list) or isinstance(data, tuple):
        data = [to_int16(x) for x in data]
    if isinstance(data, np.ndarray) and data.dtype == np.int64:
        data = data.astype(np.int16)
    return data



def modify(config, data_loader, save):
    t = time.time()
    store = data_loader.dataset.split
    # for i, data in enumerate(data_loader):
    #     data = [dict(x) for x in data]
    #
    #     out = []
    #     for j in range(len(data)):
    #         out.append(preprocess(to_long(gpu(data[j])), config['cross_dist']))
    #
    #     for j, graph in enumerate(out):
    #         idx = graph['idx']
    #         store[idx]['graph']['left'] = graph['left']
    #         store[idx]['graph']['right'] = graph['right']
    #
    #     if (i + 1) % 100 == 0:
    #         print((i + 1) * config['batch_size'], time.time() - t)
    #         t = time.time()

    f = open(os.path.join(root_path, 'SRFNet', 'dataset', 'preprocess_GAN', os.path.basename(save)), 'wb')
    pickle.dump(store, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()

class PreprocessDataset():
    def __init__(self, split, config, train=True):
        self.split = split
        self.config = config
        self.train = train

    def __getitem__(self, idx):
        from SRFNet.data import from_numpy, ref_copy
        data = self.split[idx]
        graph = dict()
        for key in ['lane_idcs', 'ctrs', 'pre_pairs', 'suc_pairs', 'left_pairs', 'right_pairs', 'feats']:
            graph[key] = ref_copy(data['graph'][key])
        graph['idx'] = idx
        return graph

    def __len__(self):
        return len(self.split)




def preprocess(graph, cross_dist, cross_angle=None):
    left, right = dict(), dict()

    lane_idcs = graph['lane_idcs']
    num_nodes = len(lane_idcs)
    num_lanes = lane_idcs[-1].item() + 1

    dist = graph['ctrs'].unsqueeze(1) - graph['ctrs'].unsqueeze(0)
    dist = torch.sqrt((dist ** 2).sum(2))
    hi = torch.arange(num_nodes).long().to(dist.device).view(-1, 1).repeat(1, num_nodes).view(-1)
    wi = torch.arange(num_nodes).long().to(dist.device).view(1, -1).repeat(num_nodes, 1).view(-1)
    row_idcs = torch.arange(num_nodes).long().to(dist.device)

    if cross_angle is not None:
        f1 = graph['feats'][hi]
        f2 = graph['ctrs'][wi] - graph['ctrs'][hi]
        t1 = torch.atan2(f1[:, 1], f1[:, 0])
        t2 = torch.atan2(f2[:, 1], f2[:, 0])
        dt = t2 - t1
        m = dt > 2 * np.pi
        dt[m] = dt[m] - 2 * np.pi
        m = dt < -2 * np.pi
        dt[m] = dt[m] + 2 * np.pi
        mask = torch.logical_and(dt > 0, dt < config['cross_angle'])
        left_mask = mask.logical_not()
        mask = torch.logical_and(dt < 0, dt > -config['cross_angle'])
        right_mask = mask.logical_not()

    pre = graph['pre_pairs'].new().float().resize_(num_lanes, num_lanes).zero_()
    pre[graph['pre_pairs'][:, 0], graph['pre_pairs'][:, 1]] = 1
    suc = graph['suc_pairs'].new().float().resize_(num_lanes, num_lanes).zero_()
    suc[graph['suc_pairs'][:, 0], graph['suc_pairs'][:, 1]] = 1

    pairs = graph['left_pairs']
    if len(pairs) > 0:
        mat = pairs.new().float().resize_(num_lanes, num_lanes).zero_()
        mat[pairs[:, 0], pairs[:, 1]] = 1
        mat = (torch.matmul(mat, pre) + torch.matmul(mat, suc) + mat) > 0.5

        left_dist = dist.clone()
        mask = mat[lane_idcs[hi], lane_idcs[wi]].logical_not()
        left_dist[hi[mask], wi[mask]] = 1e6
        if cross_angle is not None:
            left_dist[hi[left_mask], wi[left_mask]] = 1e6

        min_dist, min_idcs = left_dist.min(1)
        mask = min_dist < cross_dist
        ui = row_idcs[mask]
        vi = min_idcs[mask]
        f1 = graph['feats'][ui]
        f2 = graph['feats'][vi]
        t1 = torch.atan2(f1[:, 1], f1[:, 0])
        t2 = torch.atan2(f2[:, 1], f2[:, 0])
        dt = torch.abs(t1 - t2)
        m = dt > np.pi
        dt[m] = torch.abs(dt[m] - 2 * np.pi)
        m = dt < 0.25 * np.pi

        ui = ui[m]
        vi = vi[m]

        left['u'] = ui.cpu().numpy().astype(np.int16)
        left['v'] = vi.cpu().numpy().astype(np.int16)
    else:
        left['u'] = np.zeros(0, np.int16)
        left['v'] = np.zeros(0, np.int16)

    pairs = graph['right_pairs']
    if len(pairs) > 0:
        mat = pairs.new().float().resize_(num_lanes, num_lanes).zero_()
        mat[pairs[:, 0], pairs[:, 1]] = 1
        mat = (torch.matmul(mat, pre) + torch.matmul(mat, suc) + mat) > 0.5

        right_dist = dist.clone()
        mask = mat[lane_idcs[hi], lane_idcs[wi]].logical_not()
        right_dist[hi[mask], wi[mask]] = 1e6
        if cross_angle is not None:
            right_dist[hi[right_mask], wi[right_mask]] = 1e6

        min_dist, min_idcs = right_dist.min(1)
        mask = min_dist < cross_dist
        ui = row_idcs[mask]
        vi = min_idcs[mask]
        f1 = graph['feats'][ui]
        f2 = graph['feats'][vi]
        t1 = torch.atan2(f1[:, 1], f1[:, 0])
        t2 = torch.atan2(f2[:, 1], f2[:, 0])
        dt = torch.abs(t1 - t2)
        m = dt > np.pi
        dt[m] = torch.abs(dt[m] - 2 * np.pi)
        m = dt < 0.25 * np.pi

        ui = ui[m]
        vi = vi[m]

        right['u'] = ui.cpu().numpy().astype(np.int16)
        right['v'] = vi.cpu().numpy().astype(np.int16)
    else:
        right['u'] = np.zeros(0, np.int16)
        right['v'] = np.zeros(0, np.int16)

    out = dict()
    out['left'] = left
    out['right'] = right
    out['idx'] = graph['idx']
    return out



def to_long(data):
    if isinstance(data, dict):
        for key in data.keys():
            data[key] = to_long(data[key])
    if isinstance(data, list) or isinstance(data, tuple):
        data = [to_long(x) for x in data]
    if torch.is_tensor(data) and data.dtype == torch.int16:
        data = data.long()
    return data


def gpu(data):
    """
    Transfer tensor in `data` to gpu recursively
    `data` can be dict, list or tuple
    """
    if isinstance(data, list) or isinstance(data, tuple):
        data = [gpu(x) for x in data]
    elif isinstance(data, dict):
        data = {key:gpu(_data) for key,_data in data.items()}
    elif isinstance(data, torch.Tensor):
        data = data.contiguous().cuda(non_blocking=True)
    return data


def reform(ego_fut_traj, cl_cands_target, init_pred_global):

    i = 0
    if len(cl_cands_target) == 1:
        cl_cand = torch.from_numpy(cl_cands_target[i])
    else:
        cl_cand = get_nearest_cl([torch.from_numpy(cl_cands_target[j]) for j in range(len(cl_cands_target))], init_pred_global)
    cl_cand = cl_cand[:100]
    ego_fut = torch.from_numpy(ego_fut_traj[i])
    mask = torch.zeros_like(ego_fut)
    mask = torch.repeat_interleave(torch.repeat_interleave(mask, 4, dim=0)[:100, 0].unsqueeze(dim=0), 30, dim=0)
    ego_pos = ego_fut
    ego_dists = torch.norm(torch.repeat_interleave(cl_cand.unsqueeze(dim=1), 30, dim=1) - ego_pos, dim=2)
    ego_dist, ego_idx = torch.min(ego_dists, dim=0)

    sur_pos = init_pred_global[0, :, :, :]
    sur_dists = [torch.norm(torch.repeat_interleave(cl_cand.unsqueeze(dim=1), 30, dim=1) - sur_pos[i, :], dim=2) for i in range(6)]
    sur_dist, sur_idx = [], []
    for j in range(6):
        sur_dist_tmp, sur_idx_tmp = torch.min(sur_dists[j], dim=0)
        sur_dist.append(sur_dist_tmp)
        sur_idx.append(sur_idx_tmp)
    sur_min_idx = min([min(sur_idx[ss]) for ss in range(6)])
    min_idx = min(sur_min_idx, min(ego_idx))

    sur_disp = [cl_cand[sur_idx[i]] - sur_pos[i] for i in range(6)]
    sur_feat_1 = [mask.clone() for _ in range(6)]
    for jj in range(6):
        sur_feat_1[jj][np.arange(30), sur_idx[jj] - min_idx] = 1
    sur_feat = torch.cat([torch.cat([sur_feat_1[i], sur_disp[i]], dim=1).unsqueeze(dim=0) for i in range(6)], dim=0)

    ego_disp = cl_cand[ego_idx] - ego_pos
    ego_feat_1 = mask.clone()
    ego_feat_1[np.arange(30), ego_idx - min_idx] = 1
    ego_feat = torch.repeat_interleave(torch.cat([ego_feat_1, ego_disp], dim=1).unsqueeze(dim=0), 6, dim=0)
    feat = torch.cat([ego_feat, sur_feat], dim=-1)

    return feat

def get_nearest_cl(cl_cands_target_tmp, init_pred_global_tmp):
    dist = []
    for i in range(len(cl_cands_target_tmp)):
        dist_tmp = []
        cl_tmp = cl_cands_target_tmp[i]
        cl_tmp_dense = get_cl_dense(cl_tmp)
        for j in range(30):
            tmp = cl_tmp_dense - init_pred_global_tmp[0, :, j:j + 1, :]
            tmps = torch.sqrt(tmp[:, :, 0] ** 2 + tmp[:, :, 1] ** 2)
            dist_tmp_tmp = torch.mean(torch.min(tmps, dim=1)[0]).unsqueeze(dim=0)
            dist_tmp.append(dist_tmp_tmp)
        dist_tmp = torch.cat(dist_tmp)
        dist.append(torch.mean(dist_tmp).unsqueeze(dim=0))
    dist = torch.cat(dist)
    return cl_cands_target_tmp[torch.argmin(dist)]

def get_cl_dense(cl_tmp):
    cl_mod = torch.zeros_like(torch.repeat_interleave(cl_tmp, 4, dim=0))
    for i in range(cl_tmp.shape[0]):
        if i == cl_tmp.shape[0] - 1:
            cl_mod[4 * i, :] = cl_tmp[i, :]
        else:
            cl_mod[4 * i, :] = cl_tmp[i, :]
            cl_mod[4 * i + 1, :] = cl_tmp[i, :] + 1 * (cl_tmp[i + 1, :] - cl_tmp[i, :]) / 4
            cl_mod[4 * i + 2, :] = cl_tmp[i, :] + 2 * (cl_tmp[i + 1, :] - cl_tmp[i, :]) / 4
            cl_mod[4 * i + 3, :] = cl_tmp[i, :] + 3 * (cl_tmp[i + 1, :] - cl_tmp[i, :]) / 4
    cl_mod = cl_mod[:-3, :]
    return torch.repeat_interleave(cl_mod.unsqueeze(dim=0), 6, dim=0)


if __name__ == "__main__":
    main()
