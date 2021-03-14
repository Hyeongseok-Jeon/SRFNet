import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import pickle
import os


class TrajectoryDataset(Dataset):
    def __init__(self, meta_file_dir, root_dir):
        self.meta = pd.read_csv(meta_file_dir, header=None, prefix="var")
        self.root_dir = root_dir

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
                  'nodes': data['nodes'],
                  'graph_idcs': data['graph_idcs'],
                  'ego_feat': data['ego_feats'],
                  'file_name': data['file_name']}

        return sample

def batch_form(samples):
    actors = np.concatenate([sample['actors'] for sample in samples], dim=0)
    actor_idcs = [sample['actor_idcs'] for sample in samples]
    actor_ctrs = [sample['actor_ctrs'] for sample in samples]
    file_name = [sample['file_name'] for sample in samples]


    return {'fut_traj': torch.from_numpy(fut_traj_batch),
            'ref_path': torch.from_numpy(ref_path_batch),
            'ref_maneuver': torch.from_numpy(ref_maneuver_batch),
            'file_name': file_name_lists,
            'nan_file_name': nan_file_name_lists}