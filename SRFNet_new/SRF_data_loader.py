import torch
import glob
import pickle5 as pickle

class SRF_data_loader(torch.utils.data.Dataset):
    def __init__(self, config, train):
        if train:
            self.root_dir = config['SRF_data_train_dir']
        else:
            self.root_dir = config['SRF_data_val_dir']

        self.data_list = glob.glob(self.root_dir + '/*.pickle')

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        with open(self.data_list[idx], 'rb') as f:
            data = pickle.load(f)
        return [data[0], data[1], data[2]]


def collate_fn(batch):
    max_len = max([batch[i][0].shape[2] for i in range(len(batch))])
    mask = torch.zeros(size=[len(batch), 14, max_len, 50, 30, 2])
    for i in range(len(batch)):
        mask[i, :, :batch[i][0].shape[2], :, :, :] = batch[i][0]

    action_input_tot = torch.cat([batch[i][1] for i in range(len(batch))], dim=0)
    graphs = [batch[i][2][0] for i in range(len(batch))]

    return mask, action_input_tot, graphs

