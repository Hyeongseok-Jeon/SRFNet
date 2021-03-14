import warnings
import os
import argparse
import numpy as np
import sys

sys.path.extend(['/home/jhs/Desktop/SRFNet'])
sys.path.extend(['/home/jhs/Desktop/SRFNet/LaneGCN'])
sys.path.extend(['/home/user/data/HyeongseokJeon/infogan_pred/SRFNet'])
sys.path.extend(['/home/user/data/HyeongseokJeon/infogan_pred/SRFNet/LaneGCN'])
import time
import torch
from torch.utils.data import DataLoader
from SRFNet.data import ArgoDataset as Dataset, collate_fn
from LaneGCN.lanegcn import PostProcess, pred_metrics
from SRFNet.config import get_config
from LaneGCN.utils import Optimizer
from SRFNet.model import Net_min, Loss

warnings.filterwarnings("ignore")

root_path = os.path.join(os.path.abspath(os.curdir))
sys.path.insert(0, root_path)

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--memo', type=str, default='')
parser.add_argument('--location', type=str, default='home')
parser.add_argument('--pre', type=bool, default=False)
parser.add_argument("--mode", default='client')
parser.add_argument("--port", default=52162)
args = parser.parse_args()


def main():
    config = get_config(root_path, args)
    config['gpu_id'] = args.gpu_id
    config["save_dir"] = config["save_dir"] + '_'+args.memo
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
                            batch_size=config["val_batch_size"],
                            num_workers=config["val_workers"],
                            collate_fn=collate_fn,
                            shuffle=True,
                            pin_memory=True)

    net = Net_min(config)
    pre_trained_weight = torch.load(os.path.join(root_path, "LaneGCN/pre_trained") + '/36.000.ckpt')
    pretrained_dict = pre_trained_weight['state_dict']
    new_model_dict = net.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in new_model_dict}
    new_model_dict.update(pretrained_dict)
    net.load_state_dict(new_model_dict)
    net = net.cuda(config['gpu_id'])

    opt = Optimizer(net.parameters(), config)
    loss = Loss(config)

    train(config, train_loader, net, loss, post_process, opt, val_loader)


def train(config, train_loader, net, loss, post_process, opt, val_loader=None):
    net.train()

    display_iters = config["display_iters"]
    val_iters = config["val_iters"]

    start_time = time.time()
    metrics = dict()
    batch_num = len(train_loader.dataset)
    for epoch in range(config['num_epochs']):
        update_num = 0
        ade1_tot = 0
        fde1_tot = 0
        ade_tot = 0
        fde_tot = 0
        loss_tot = 0
        init_time = time.time()
        for i, data in enumerate(train_loader):
            current = (i + 1) * config['batch_size']
            percent = float(current) * 100 / batch_num
            arrow = '-' * int(percent / 100 * 20 - 1) + '>'
            spaces = ' ' * (20 - len(arrow))
            if i == 0:
                sys.stdout.write('\n' + ' %d th Epoch Progress: [%s%s] %d %%  time: %f sec' % (epoch + 1, arrow, spaces, percent, time.time() - init_time))
            else:
                sys.stdout.write('\r' + ' %d th Epoch Progress: [%s%s] %d %%  time: %f sec    [loss: %f] [ade1: %f] [fde1: %f] [ade: %f] [fde: %f]' % (
                    epoch + 1, arrow, spaces, percent, time.time() - init_time, loss_tot / update_num, ade1_tot / update_num, fde1_tot / update_num, ade_tot / update_num, fde_tot / update_num))

            data = dict(data)
            out = net(data)
            loss_out = loss(out, data)
            post_out = post_process(out, data)
            ade1, fde1, ade, fde, _ = pred_metrics(np.concatenate(post_out['preds'], 0),
                                                   np.concatenate(post_out['gt_preds'], 0),
                                                   np.concatenate(post_out['has_preds'], 0))

            ade1_tot += ade1 * len(data["city"])
            fde1_tot += fde1 * len(data["city"])
            ade_tot += ade * len(data["city"])
            fde_tot += fde * len(data["city"])
            loss_tot += loss_out["loss"].item() * len(data["city"])
            update_num += len(data["city"])

            opt.zero_grad()
            loss_out["loss"].backward()
            lr = opt.step(epoch)

        if epoch % display_iters == display_iters - 1:
            dt = time.time() - start_time
            post_process.display(metrics, dt, epoch, lr)
            start_time = time.time()
            metrics = dict()

        if epoch % val_iters == val_iters - 1:
            save_ckpt(net, opt, config["save_dir"], epoch)
            val(config, val_loader, net, loss, post_process, epoch)


def val(config, data_loader, net, loss, post_process, epoch):
    net.eval()
    start_time = time.time()
    metrics = dict()
    for i, data in enumerate(data_loader):
        data = dict(data)
        with torch.no_grad():
            _, output_non_interact, output_sur_interact, output_ego_interact = net(data)
            loss_out = loss(output_non_interact, data)
            post_out = post_process(output_non_interact, data)
            post_process.append(metrics, loss_out, post_out)

    dt = time.time() - start_time
    post_process.display(metrics, dt, epoch)
    net.train()


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


if __name__ == "__main__":
    main()
