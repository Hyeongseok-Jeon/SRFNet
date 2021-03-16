import warnings
import os
import argparse
import numpy as np
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
from SRFNet.data_SRF import TrajectoryDataset, batch_form
from LaneGCN.lanegcn import pred_metrics
from SRFNet.config import get_config
from LaneGCN.utils import Optimizer, gpu, cpu
from SRFNet.model import Net_min, Loss, Net, Loss_light, PostProcess
import pickle5 as pickle
from torch.utils.tensorboard import SummaryWriter
import horovod.torch as hvd

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
parser.add_argument("--multi_gpu", type=bool, default=True)
parser.add_argument("--interaction", type=str, default='ego')
args = parser.parse_args()


def main():
    config = get_config(root_path, args)
    config['gpu_id'] = args.gpu_id
    config["save_dir"] = config["save_dir"] + '_' + args.memo
    # post processing function
    post_process = PostProcess(config)

    if args.location == 'home':
        debug_dataset = TrajectoryDataset(config["val_meta"], config["data_root"] + 'val/', config)
        debug_loader = DataLoader(debug_dataset,
                                  batch_size=config["val_batch_size"],
                                  num_workers=config["val_workers"],
                                  collate_fn=batch_form,
                                  shuffle=True,
                                  pin_memory=True)
    else:
        # data loader for training
        train_dataset = TrajectoryDataset(config["train_meta"], config["data_root"] + 'train/', config)
        train_loader = DataLoader(train_dataset,
                                  batch_size=config["batch_size"],
                                  num_workers=config["workers"],
                                  collate_fn=batch_form,
                                  shuffle=True,
                                  pin_memory=True)
        val_dataset = TrajectoryDataset(config["val_meta"], config["data_root"] + 'val/', config)
        val_loader = DataLoader(val_dataset,
                                batch_size=config["val_batch_size"],
                                num_workers=config["val_workers"],
                                collate_fn=batch_form,
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
    loss = Loss_light(config)
    if args.multi_gpu:
        hvd.init()
        torch.cuda.set_device(hvd.local_rank())
        if args.location == 'home':
            debug_sampler = torch.utils.data.distributed.DistributedSampler(
                debug_dataset, num_replicas=hvd.size(), rank=hvd.rank())
            debug_loader = torch.utils.data.DataLoader(debug_dataset,
                                                       batch_size=config["batch_size"],
                                                       num_workers=config["workers"],
                                                       collate_fn=batch_form,
                                                       pin_memory=True,
                                                       sampler=debug_sampler)
            net = net.cuda()
            opt.opt = hvd.DistributedOptimizer(opt.opt, named_parameters=net.named_parameters())
            hvd.broadcast_parameters(net.state_dict(), root_rank=0)
    else:
        net = net.cuda(config['gpu_id'])

    if args.location == 'home':
        train(config, debug_loader, net, loss, post_process, opt, debug_loader)
    else:
        train(config, train_loader, net, loss, post_process, opt, val_loader)


def train(config, train_loader, net, loss, post_process, opt, val_loader=None):
    net.train()
    val_iters = config["val_iters"]
    save_iters = config['save_freq']
    batch_num = len(train_loader.dataset)
    first_val = True
    writer = SummaryWriter(config["save_dir"] + '/tensorboard')
    for epoch in range(config['num_epochs']):
        update_num = 0
        ade1_tot = 0
        fde1_tot = 0
        ade_tot = 0
        fde_tot = 0
        loss_tot = 0
        init_time = time.time()
        time_ref = 0
        for i, data in enumerate(train_loader):
            net.zero_grad()
            current = (i + 1) * config['batch_size']
            percent = float(current) * 100 / batch_num
            arrow = '-' * int(percent / 100 * 20 - 1) + '>'
            spaces = ' ' * (20 - len(arrow))
            if i == 0:
                sys.stdout.write('\n' + ' %d th Epoch Progress: [%s%s] %d %%  time: %f sec' % (epoch + 1, arrow, spaces, percent, time.time() - init_time))
            else:
                sys.stdout.write('\r' + ' %d th Epoch Progress: [%s%s] %d %%  time: %f sec    [loss: %f] [ade1: %f] [fde1: %f] [ade: %f] [fde: %f]' % (
                    epoch + 1, arrow, spaces, percent, time.time() - init_time, loss_tot / update_num, ade1_tot / update_num, fde1_tot / update_num, ade_tot / update_num, fde_tot / update_num))

            actor_ctrs = gpu(data['actor_ctrs'], gpu_id=config['gpu_id'])
            actor_idcs = gpu(data['actor_idcs'], gpu_id=config['gpu_id'])
            actors = gpu(data['actors'], gpu_id=config['gpu_id'])
            nodes = gpu(data['nodes'], gpu_id=config['gpu_id'])
            graph_idcs = gpu(data['graph_idcs'], gpu_id=config['gpu_id'])
            ego_feat = gpu(data['ego_feat'], gpu_id=config['gpu_id'])
            feats = gpu(data['feats'], gpu_id=config['gpu_id'])
            nearest_ctrs_hist = gpu(data['nearest_ctrs_hist'], gpu_id=config['gpu_id'])
            rot = gpu(data['rot'], gpu_id=config['gpu_id'])
            orig = gpu(data['orig'], gpu_id=config['gpu_id'])
            gt_preds = gpu(data['gt_preds'], gpu_id=config['gpu_id'])
            has_preds = gpu(data['has_preds'], gpu_id=config['gpu_id'])
            ego_feat_calc = gpu(data['ego_feat_calc'], gpu_id=config['gpu_id'])
            inputs = [actor_ctrs, actor_idcs, actors, nodes, graph_idcs, ego_feat, feats, nearest_ctrs_hist, rot, orig, ego_feat_calc]

            out = net(inputs)
            loss_out = loss(out, gt_preds, has_preds)
            post_out = post_process(out, gt_preds, has_preds)
            ade1, fde1, ade, fde, _ = pred_metrics(np.concatenate(post_out['preds'], 0),
                                                   np.concatenate(post_out['gt_preds'], 0),
                                                   np.concatenate(post_out['has_preds'], 0))

            ade1_tot += ade1 * len(data["city"])
            fde1_tot += fde1 * len(data["city"])
            ade_tot += ade * len(data["city"])
            fde_tot += fde * len(data["city"])
            loss_tot += loss_out["loss"].item() * len(data["city"])
            update_num += len(data["city"])

            loss_out["loss"].backward()
            lr = opt.step(epoch)

        if epoch % val_iters == val_iters - 1:
            if not os.path.exists(config["save_dir"]):
                os.makedirs(config["save_dir"])
            if first_val:
                with open(config["save_dir"] + '/info.pickle', 'wb') as f:
                    pickle.dump([args, config], f, pickle.HIGHEST_PROTOCOL)
                first_val = False
            [loss_val, ade1_val, fde1_val, ade6_val, fde6_val] = val(config, val_loader, net, loss, post_process, epoch)
            writer.add_scalar('loss_val', loss_val, epoch)
            writer.add_scalar('ade_val', ade1_val, epoch)
            writer.add_scalar('fde_val', fde1_val, epoch)
            writer.add_scalar('ade6_val', ade6_val, epoch)
            writer.add_scalar('fde6_val', fde6_val, epoch)
        if epoch % save_iters == save_iters - 1:
            if not os.path.exists(config["save_dir"]):
                os.makedirs(config["save_dir"])
            save_ckpt(net, opt, config["save_dir"], epoch)
        writer.add_scalar('loss_train', loss_tot / update_num, epoch)
        writer.add_scalar('ade_train', ade1_tot / update_num, epoch)
        writer.add_scalar('fde_train', fde1_tot / update_num, epoch)
        writer.add_scalar('ade6_train', ade_tot / update_num, epoch)
        writer.add_scalar('fde6_train', fde_tot / update_num, epoch)


def val(config, data_loader, net, loss, post_process, epoch):
    net.eval()
    update_num = 0
    ade1_tot = 0
    fde1_tot = 0
    ade_tot = 0
    fde_tot = 0
    loss_tot = 0
    batch_num = len(data_loader.dataset)
    init_time = time.time()

    for i, data in enumerate(data_loader):
        current = (i + 1) * config['batch_size']
        percent = float(current) * 100 / batch_num
        arrow = '-' * int(percent / 100 * 20 - 1) + '>'
        spaces = ' ' * (20 - len(arrow))
        if i == 0:
            sys.stdout.write('\n' + ' Validation Progress: [%s%s] %d %%  time: %f sec' % (arrow, spaces, percent, time.time() - init_time))

        data = dict(data)
        actor_ctrs = gpu(data['actor_ctrs'], gpu_id=config['gpu_id'])
        actor_idcs = gpu(data['actor_idcs'], gpu_id=config['gpu_id'])
        actors = gpu(data['actors'], gpu_id=config['gpu_id'])
        nodes = gpu(data['nodes'], gpu_id=config['gpu_id'])
        graph_idcs = gpu(data['graph_idcs'], gpu_id=config['gpu_id'])
        ego_feat = gpu(data['ego_feat'], gpu_id=config['gpu_id'])
        feats = gpu(data['feats'], gpu_id=config['gpu_id'])
        nearest_ctrs_hist = gpu(data['nearest_ctrs_hist'], gpu_id=config['gpu_id'])
        rot = gpu(data['rot'], gpu_id=config['gpu_id'])
        orig = gpu(data['orig'], gpu_id=config['gpu_id'])
        gt_preds = gpu(data['gt_preds'], gpu_id=config['gpu_id'])
        has_preds = gpu(data['has_preds'], gpu_id=config['gpu_id'])
        ego_feat_calc = gpu(data['ego_feat_calc'], gpu_id=config['gpu_id'])

        with torch.no_grad():
            inputs = [actor_ctrs, actor_idcs, actors, nodes, graph_idcs, ego_feat, feats, nearest_ctrs_hist, rot, orig, ego_feat_calc]
            out = net(inputs)
            loss_out = loss(out, gt_preds, has_preds)
            post_out = post_process(out, gt_preds, has_preds)
            ade1, fde1, ade, fde, _ = pred_metrics(np.concatenate(post_out['preds'], 0),
                                                   np.concatenate(post_out['gt_preds'], 0),
                                                   np.concatenate(post_out['has_preds'], 0))
            ade1_tot += ade1 * len(data["city"])
            fde1_tot += fde1 * len(data["city"])
            ade_tot += ade * len(data["city"])
            fde_tot += fde * len(data["city"])
            loss_tot += loss_out["loss"].item() * len(data["city"])
            update_num += len(data["city"])
    sys.stdout.write('\r' + ' Validation is completed: [loss: %f] [ade1: %f] [fde1: %f] [ade: %f] [fde: %f]' % (
        loss_tot / update_num, ade1_tot / update_num, fde1_tot / update_num, ade_tot / update_num, fde_tot / update_num))

    net.train()
    return [loss_tot / update_num, ade1_tot / update_num, fde1_tot / update_num, ade_tot / update_num, fde_tot / update_num]


def save_ckpt(net, opt, save_dir, epoch):
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
