import warnings
import os
import argparse
import sys

sys.path.extend(['/home/jhs/Desktop/SRFNet'])
sys.path.extend(['/home/jhs/Desktop/SRFNet/LaneGCN'])
sys.path.extend(['/home/user/Desktop/SRFNet'])
sys.path.extend(['/home/user/Desktop/SRFNet/LaneGCN'])
sys.path.extend(['/home/user/data/HyeongseokJeon/infogan_pred/SRFNet'])
sys.path.extend(['/home/user/data/HyeongseokJeon/infogan_pred/SRFNet/LaneGCN'])

import time
import torch
from torch.utils.data import DataLoader
from SRFNet.data import ArgoDataset as Dataset, collate_fn
from LaneGCN.lanegcn import PostProcess, pred_metrics
from SRFNet.config import get_config
from LaneGCN.utils import Optimizer, cpu
from SRFNet.model import pre_net, Loss
import pickle

warnings.filterwarnings("ignore")

root_path = os.path.join(os.path.abspath(os.curdir))
sys.path.insert(0, root_path)

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--memo', type=str, default='')
parser.add_argument('--location', type=str, default='home')
parser.add_argument('--pre', type=bool, default=True)
parser.add_argument("--mode", default='client')
parser.add_argument("--port", default=52162)
args = parser.parse_args()


def main():
    config = get_config(root_path, args)
    config['gpu_id'] = args.gpu_id
    config["save_dir"] = config["save_dir"] + '_'+args.memo

    data_root, _ = os.path.split(config['preprocess_train'])
    train_path = data_root + '/train'
    val_path = data_root + '/val'
    test_path = data_root + '/test'
    try:
        os.mkdir(train_path)
        os.mkdir(val_path)
        os.mkdir(test_path)
    except:
        pass

    # post processing function
    post_process = PostProcess(config)

    # data loader for training
    dataset = Dataset(config["train_split"], config, train=True)
    train_loader = DataLoader(dataset,
                              batch_size=config["batch_size"],
                              num_workers=config["workers"],
                              collate_fn=collate_fn,
                              shuffle=True,
                              pin_memory=True,
                              drop_last=True)

    # Data loader for evaluation
    dataset = Dataset(config["val_split"], config, train=False)
    val_loader = DataLoader(dataset,
                            batch_size=config["batch_size"],
                            num_workers=config["val_workers"],
                            collate_fn=collate_fn,
                            shuffle=True,
                            pin_memory=True)

    net = pre_net(config)
    pre_trained_weight = torch.load(os.path.join(root_path, "LaneGCN/pre_trained") + '/36.000.ckpt')
    pretrained_dict = pre_trained_weight['state_dict']
    new_model_dict = net.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in new_model_dict}
    new_model_dict.update(pretrained_dict)
    net.load_state_dict(new_model_dict)
    net = net.cuda(config['gpu_id'])

    opt = Optimizer(net.parameters(), config)
    loss = Loss(config)

    # data_gen(config, val_loader, net, val_path)
    data_gen(config, train_loader, net, train_path)


def data_gen(config, train_loader, net, path):
    batch_num = len(train_loader.dataset)
    init_time = time.time()
    for i, data in enumerate(train_loader):
        current = (i + 1) * config['batch_size']
        percent = float(current) * 100 / batch_num
        arrow = '-' * int(percent / 100 * 20 - 1) + '>'
        spaces = ' ' * (20 - len(arrow))
        if i == 0:
            sys.stdout.write('\n' + 'Progress: [%s%s] %d %%  time: %f sec  %s '% (arrow, spaces, percent, time.time() - init_time,  str(data['file_name'][0])))
        else:
            sys.stdout.write('\r' + 'Progress: [%s%s] %d %%  time: %f sec  %s '% (arrow, spaces, percent, time.time() - init_time,  str(data['file_name'][0])))
        _, file_name = os.path.split(data['file_name'][0])
        file_name = file_name[:-4]
        # if '45638' in str(data['file_name'][0]):
        #     print('got it')
        #     time.sleep(100)
        try:
            [actors_hidden, nodes, node_idcs, node_ctrs, graph_idcs] = net(data)
        except:
            with open(path + '/' + file_name + 'err.pickle', 'wb') as f:
                pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
            pass
        # 131251
        data['actors_hidden'] = cpu([x.detach() for x in actors_hidden])
        data['node'] = cpu(nodes.detach())
        data['node_idcs'] = cpu([x.detach() for x in node_idcs])
        data['node_ctrs'] = cpu([x.detach() for x in node_ctrs])
        data['graph_idcs'] = cpu([x.detach() for x in graph_idcs])

        with open(path + '/'+file_name+'.pickle', 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
