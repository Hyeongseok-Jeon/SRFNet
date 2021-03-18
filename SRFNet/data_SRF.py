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
        actors, _ = self.actor_gather(data["feats"][0])

        sample = {'actors': actors,
                  'actor_ctrs': data['ctrs'],
                  'actor_idcs': data['actor_idcs'],
                  'actors_hidden': data['actors_hidden'],
                  'nodes': data['node'],
                  'graph_idcs': data['graph_idcs'],
                  'ego_feat': data['ego_feats'],
                  'nearest_ctrs_hist': data['nearest_ctrs_hist'],
                  'feats': data['feats'],
                  'file_name': data['file_name'],
                  'rot': data['rot'],
                  'orig': data['orig'],
                  'gt_preds': data['gt_preds'],
                  'has_preds': data['has_preds'],
                  'ego_feat_calc': data['ego_feat_calc'],
                  'graph_mod': data['graph_mod'],
                  'city': data['city']}
        # sample = self.stop_filter(sample)

        return sample

    def stop_filter(self, sample):
        displacement = sample['feats'][0]
        displacements = torch.norm(torch.sum(displacement, dim=1)[:,:2], dim=1)

        stop_idx_bool = displacements > 1
        stop_idx_bool[0] = True
        stop_idx_bool[1] = True
        stop_idx = torch.where(stop_idx_bool)[0]

        sample['actor_ctrs'][0] = sample['actor_ctrs'][0][stop_idx]
        sample['actor_idcs'][0][0] = sample['actor_idcs'][0][0][:len(stop_idx)]
        sample['actors_hidden'][0] = sample['actors_hidden'][0][stop_idx]
        sample['nearest_ctrs_hist'][0] = sample['nearest_ctrs_hist'][0][stop_idx]
        sample['feats'][0] = sample['feats'][0][stop_idx]
        sample['gt_preds'][0] = sample['gt_preds'][0][stop_idx]
        sample['has_preds'][0] = sample['has_preds'][0][stop_idx]
        sample['ego_feat_calc'] = [sample['ego_feat_calc'][i][stop_idx] for i in range(30)]

        return sample

    def actor_gather(self, actors):
        num_actors = len(actors)
        actors_time = []
        for j in range(20):
            zero_pad = torch.zeros_like(actors)[:, j + 1:, :]
            tmp = actors[:, :j + 1, :]
            actors_time.append(torch.transpose(torch.cat((zero_pad, tmp), dim=1), 2, 1))

        actors = torch.cat(actors_time, dim=0)

        actor_idcs = []
        idcs = np.arange(0, num_actors)
        actor_idcs.append(idcs)

        return actors, actor_idcs


def batch_form(samples):
    actors = [sample['actors'] for sample in samples]
    actors_hidden = [sample['actors_hidden'][0] for sample in samples]
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
    ego_feat_calc =[sample['ego_feat_calc'] for sample in samples]
    graph_mod =[sample['graph_mod'][0] for sample in samples]

    ego_feat_calc_mod = []
    for i in range(30):
        tmp = []
        for j in range(len(ego_feat_calc)):
            tmp.append(ego_feat_calc[j][i])
        ego_feat_calc_mod.append(torch.cat(tmp, dim = 0))

    city = [sample['city'] for sample in samples]

    sample_mod = {'actors': actors,
                  'actor_ctrs': actor_ctrs,
                  'actor_idcs': actor_idcs,
                  'actors_hidden': actors_hidden,
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
                  'ego_feat_calc': ego_feat_calc_mod,
                  'graph_mod': graph_mod,
                  'city': city}

    return sample_mod
