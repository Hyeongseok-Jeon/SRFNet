import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import pickle5 as pickle
import os
from LaneGCN.utils import gpu

class TrajectoryDataset(Dataset):
    def __init__(self, meta_file_dir, root_dir, config):
        self.meta = pd.read_csv(meta_file_dir, header=None, prefix="var")
        self.root_dir = root_dir
        self.config = config

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data_path = os.path.join(self.root_dir)
        data_name = os.path.join(data_path, self.meta.iloc[idx, 0])
        with open(data_name, 'rb') as f:
            data = pickle.load(f)

        sample = {'actor_ctrs': data['ctrs'],
                  'actor_idcs': data['actor_idcs'],
                  'actors': data['actors_hidden'],
                  'nodes': data['node'],
                  'graph_idcs': data['graph_idcs'],
                  'ego_feat': data['ego_feats'],
                  'nearest_ctrs_hist': data['nearest_ctrs_hist'],
                  'feats' : data['feats'],
                  'file_name': data['file_name'],
                  'rot': data['rot'],
                  'orig': data['orig'],
                  'gt_preds': data['gt_preds'],
                  'has_preds': data['has_preds'],
                  'ego_feat_calc': data['ego_feat_calc'],
                  'city': data['city']}
        return sample


def batch_form(samples):
    actors = [sample['actors'][0] for sample in samples]
    actor_idcs = [sample['actor_idcs'][0][0] for sample in samples]
    actor_ctrs = [sample['actor_ctrs'][0] for sample in samples]
    file_name = [sample['file_name'][0] for sample in samples]
    nodes = torch.cat([sample['nodes'] for sample in samples], dim=0)
    graph_idcs = [sample['graph_idcs'][0] for sample in samples]
    ego_feats = [sample['ego_feat'][0] for sample in samples]
    nearest_ctrs_hist = [sample['nearest_ctrs_hist'][0] for sample in samples]
    feats = [sample['feats'][0] for sample in samples]
    rot = [sample['rot'] for sample in samples]
    orig = [sample['orig'] for sample in samples]
    gt_preds = [sample['gt_preds'][0] for sample in samples]
    has_preds = [sample['has_preds'][0] for sample in samples]
    ego_feat_calc = [sample['ego_feat_calc'][0] for sample in samples]
    city = [sample['city'] for sample in samples]

    sample_mod = {'actor_ctrs': actor_ctrs,
                  'actor_idcs': actor_idcs,
                  'actors': actors,
                  'nodes': nodes,
                  'graph_idcs': graph_idcs,
                  'ego_feat': ego_feats,
                  'file_name': file_name,
                  'feats': feats,
                  'nearest_ctrs_hist': nearest_ctrs_hist,
                  'rot': rot,
                  'orig': orig,
                  'gt_preds': gt_preds,
                  'has_preds': has_preds,
                  'ego_feat_calc': ego_feat_calc,
                  'city': city}

    return sample_mod
