
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
from LaneGCN_original.LaneGCN.lanegcn import get_model
import warnings
from SRFNet_new.utils import Logger, load_pretrain, gpu
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

config, _, _, net, _, _, _ = get_model()
pre_trained_weight = torch.load(os.path.join(root_path, "LaneGCN/pre_trained") + '/36.000.ckpt')
pretrained_dict = pre_trained_weight['state_dict']
new_model_dict = net.state_dict()
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in new_model_dict}
new_model_dict.update(pretrained_dict)
net.load_state_dict(new_model_dict)
os.makedirs(os.path.join(root_path, 'SRFNet_new', 'dataset', 'preprocess_GAN'), exist_ok=True)
from SRFNet_new.data import ArgoDataset as Dataset, from_numpy, ref_copy, collate_fn



def main():
    # Import all settings for experiment.
    config["preprocess"] = False  # we use raw data to generate preprocess data
    config["batch_size"] = 12
    config["workers"] = 0
    config['cross_dist'] = 6
    config['cross_angle'] = 0.5 * np.pi
    config["train_split"] = os.path.join(root_path, 'LaneGCN','dataset','train','data')
    config["val_split"] = os.path.join(root_path, 'LaneGCN','dataset','val','data')
    val(config)
    # test(config)
    train(config)


def train(config):
    # Data loader for training set
    dataset = Dataset(config["train_split"], config, net, train=True)
    train_loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        num_workers=config["workers"],
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False,
    )

    stores = [None for x in range(205942)]
    t = time.time()
    for i, data in enumerate(tqdm(train_loader)):
        data = dict(data)
        for j in range(len(data["idx"])):
            store = dict()
            for key in [
                "city",
                "trajs",
                "steps",
                "file_name",
                "feats",
                "ego_feats",
                "ctrs",
                "orig",
                "theta",
                "rot",
                "gt_preds",
                "has_preds",
                "idx",
                "graph",
                "cl_cands",
                "gt_cl_cands"
            ]:
                store[key] = to_numpy(data[key][j])
                if key in ["graph"]:
                    store[key] = to_int16(store[key])
            stores[store["idx"]] = store

        if (i + 1) % 100 == 0:
            print(i, time.time() - t)
            t = time.time()

    dataset = PreprocessDataset(stores, config, train=True)
    data_loader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        num_workers=config['workers'],
        shuffle=False,
        collate_fn=from_numpy,
        pin_memory=True,
        drop_last=False)

    modify(config, data_loader, config["preprocess_train"])


def val(config):
    # Data loader for validation set
    dataset = Dataset(config["val_split"], config, net, train=False)
    val_loader = DataLoader(
        dataset,
        batch_size=config["val_batch_size"],
        num_workers=config["val_workers"],
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    stores = [None for x in range(39472)]
    t = time.time()
    for i, data in enumerate(tqdm(val_loader)):
        data = dict(data)
        for j in range(len(data["idx"])):
            store = dict()
            for key in [
                "city",
                "trajs",
                "steps",
                "file_name",
                "feats",
                "ego_feats",
                "ctrs",
                "orig",
                "theta",
                "rot",
                "gt_preds",
                "has_preds",
                "idx",
                "graph",
                "cl_cands",
                "gt_cl_cands"
            ]:
                store[key] = to_numpy(data[key][j])
                if key in ["graph"]:
                    store[key] = to_int16(store[key])
            stores[store["idx"]] = store

        if (i + 1) % 100 == 0:
            print(i, time.time() - t)
            t = time.time()

    dataset = PreprocessDataset(stores, config, train=False)
    data_loader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        num_workers=config['workers'],
        shuffle=False,
        collate_fn=from_numpy,
        pin_memory=True,
        drop_last=False)

    modify(config, data_loader, config["preprocess_val"])


def test(config):
    dataset = Dataset(config["test_split"], config, train=False)
    test_loader = DataLoader(
        dataset,
        batch_size=config["val_batch_size"],
        num_workers=config["val_workers"],
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    stores = [None for x in range(78143)]

    t = time.time()
    for i, data in enumerate(tqdm(test_loader)):
        data = dict(data)
        for j in range(len(data["idx"])):
            store = dict()
            for key in [
                "idx",
                "city",
                "feats",
                "ctrs",
                "orig",
                "theta",
                "rot",
                "graph",
            ]:
                store[key] = to_numpy(data[key][j])
                if key in ["graph"]:
                    store[key] = to_int16(store[key])
            stores[store["idx"]] = store

        if (i + 1) % 100 == 0:
            print(i, time.time() - t)
            t = time.time()

    dataset = PreprocessDataset(stores, config, train=False)
    data_loader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        num_workers=0,
        shuffle=False,
        collate_fn=from_numpy,
        pin_memory=True,
        drop_last=False)

    modify(config, data_loader, config["preprocess_test"])


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
    for i, data in enumerate(data_loader):
        print(i)
        data = [dict(x) for x in data]

        out = []
        for j in range(len(data)):
            out.append(preprocess(to_long(gpu(data[j])), config['cross_dist']))

        for j, graph in enumerate(out):
            idx = graph['idx']
            store[idx]['graph']['left'] = graph['left']
            store[idx]['graph']['right'] = graph['right']

            data_in = collate_fn([store[idx].copy()])
            pred_out = net(data_in)

            ego_fut_traj = store[idx]['gt_preds'][0,:,:]
            cl_cands_target = store[idx]['cl_cands'][1]
            init_pred_global = pred_out['reg'][0].cpu()
            hid = reform(ego_fut_traj, cl_cands_target, init_pred_global)

            dict_batch = dict()
            dict_batch['cls'] = [pred_out['cls'][0].cpu().detach().numpy()]
            dict_batch['reg'] = [pred_out['reg'][0].cpu().detach().numpy()]
            init_pred_global_np = dict_batch

            store[idx]['action_input'] = hid.cpu().detach().numpy()
            store[idx]['init_pred_global'] = init_pred_global_np

        if (i + 1) % 100 == 0:
            print((i + 1) * config['batch_size'], time.time() - t)
            t = time.time()

    f = open(os.path.join(root_path, 'preprocess', save), 'wb')
    pickle.dump(store, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()


class PreprocessDataset():
    def __init__(self, split, config, train=True):
        self.split = split
        self.config = config
        self.train = train

    def __getitem__(self, idx):
        from data import from_numpy, ref_copy

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


def worker_init_fn(pid):
    np_seed = hvd.rank() * 1024 + int(pid)
    np.random.seed(np_seed)
    random_seed = np.random.randint(2 ** 32 - 1)
    random.seed(random_seed)


def reform(ego_fut_traj, cl_cands_target, init_pred_global):
    if len(cl_cands_target) == 1:
        cl_cand = torch.from_numpy(cl_cands_target[0])
    else:
        cl_cand = get_nearest_cl([torch.from_numpy(cl_cands_target[j]) for j in range(len(cl_cands_target))], init_pred_global)
    cl_cand = cl_cand[:100]
    ego_fut = torch.from_numpy(ego_fut_traj)
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
