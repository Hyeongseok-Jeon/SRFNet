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
from SRFNet.model_ordered import model_case_0, model_case_1, Loss_light, PostProcess, inter_loss
from SRFNet.model import Net_min
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
parser.add_argument("--multi_gpu", type=bool, default=False)
parser.add_argument("--interaction", type=str, default=None)
parser.add_argument("--case", type=int, default=None)
parser.add_argument("--subcase", type=int, default=None)
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
    if args.case == 0:
        net = model_case_0(config)
    elif args.case == 1:
        net = model_case_1(config)

    if args.multi_gpu:
        net = torch.nn.DataParallel(net)
    pre_trained_weight = torch.load(os.path.join(root_path, "LaneGCN/pre_trained") + '/36.000.ckpt')
    pretrained_dict = pre_trained_weight['state_dict']
    new_model_dict = net.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in new_model_dict}
    new_model_dict.update(pretrained_dict)
    net.load_state_dict(new_model_dict)
    if args.multi_gpu:
        net = net.cuda(0)
    else:
        net = net.cuda(config['gpu_id'])
    loss = Loss_light(config)

    if args.subcase == 1:
        opt = Optimizer(net.parameters(), config)
        opts = [opt]
        losses = [loss]
    elif args.subcase == 2:
        loss_delta = inter_loss(config)
        params1 = list(net.actor_net.parameters()) + list(net.pred_net.parameters())
        params2 = list(net.map_net.parameters()) + list(net.fusion_net.parameters()) + list(net.inter_pred_net.parameters())
        opt1 = Optimizer(params1, config)
        opt2 = Optimizer(params2, config)
        opts = [opt1, opt2]
        losses = [loss, loss_delta]
    elif args.subcase == 3:
        loss_delta = inter_loss(config)
        params = list(net.map_net.parameters()) + list(net.fusion_net.parameters()) + list(net.inter_pred_net.parameters())
        opt1 = Optimizer(params, config)
        opts = [opt1]
        losses = [loss_delta]

    if args.location == 'home':
        train(config, debug_loader, net, losses, post_process, opts, debug_loader)
    else:
        train(config, train_loader, net, losses, post_process, opts, val_loader)


def train(config, train_loader, net, losses, post_process, opts, val_loader=None):
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
        loss_inter_tot = 0
        init_time = time.time()
        time_ref = 0
        for i, data in enumerate(train_loader):
            current = (i + 1) * config['batch_size']
            percent = float(current) * 100 / batch_num
            arrow = '-' * int(percent / 100 * 20 - 1) + '>'
            spaces = ' ' * (20 - len(arrow))
            if i == 0:
                sys.stdout.write('\n' + ' %d th Epoch Progress: [%s%s] %d %%  time: %f sec' % (epoch + 1, arrow, spaces, percent, time.time() - init_time))
            else:
                if args.subcase == 2:
                    sys.stdout.write('\r' + ' %d th Epoch Progress: [%s%s] %d %%  time: %f sec    [loss_pred: %f] [loss_inter: %f] [ade1: %f] [fde1: %f] [ade: %f] [fde: %f]' % (
                        epoch + 1, arrow, spaces, percent, time.time() - init_time, loss_tot / update_num, loss_inter_tot / update_num, ade1_tot / update_num, fde1_tot / update_num, ade_tot / update_num, fde_tot / update_num))
                else:
                    sys.stdout.write('\r' + ' %d th Epoch Progress: [%s%s] %d %%  time: %f sec    [loss: %f] [ade1: %f] [fde1: %f] [ade: %f] [fde: %f]' % (
                        epoch + 1, arrow, spaces, percent, time.time() - init_time, loss_tot / update_num, ade1_tot / update_num, fde1_tot / update_num, ade_tot / update_num, fde_tot / update_num))

            data_sub = data.copy()
            actor_ctrs = gpu(data['actor_ctrs'], gpu_id=config['gpu_id'])
            actor_idcs = gpu(data['actor_idcs'], gpu_id=config['gpu_id'])
            actors_hidden = gpu(data['actors_hidden'], gpu_id=config['gpu_id'])
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
            actors = gpu(data['actors'], gpu_id=config['gpu_id'])
            graph_mod = gpu(data['graph_mod'], gpu_id=config['gpu_id'])

            inputs = [actor_ctrs, actor_idcs, actors_hidden, nodes, graph_idcs, ego_feat, feats, nearest_ctrs_hist, rot, orig, ego_feat_calc, actors, data, graph_mod]
            out = net(inputs)
            if args.case == 0 and args.subcase == 1:
                loss_out = losses[0](out, gt_preds, has_preds)
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

                opts[0].zero_grad()
                loss_out["loss"].backward()
                lr = opts[0].step(epoch)
            elif args.case == 1:
                out_non_interact = out[0]
                out_sur_interact = out[1]
                out_tot = out[2]
                if args.subcase == 1:
                    loss_out = losses[0](out_tot, gt_preds, has_preds)
                    post_out = post_process(out_tot, gt_preds, has_preds)
                    ade1, fde1, ade, fde, _ = pred_metrics(np.concatenate(post_out['preds'], 0),
                                                           np.concatenate(post_out['gt_preds'], 0),
                                                           np.concatenate(post_out['has_preds'], 0))

                    ade1_tot += ade1 * len(data["city"])
                    fde1_tot += fde1 * len(data["city"])
                    ade_tot += ade * len(data["city"])
                    fde_tot += fde * len(data["city"])
                    loss_tot += loss_out["loss"].item() * len(data["city"])
                    update_num += len(data["city"])

                    opts[0].zero_grad()
                    loss_out["loss"].backward()
                    lr = opts[0].step(epoch)
                elif args.subcase == 2:
                    loss_out1 = losses[0](out_non_interact, gt_preds, has_preds)
                    opts[0].zero_grad()
                    loss_out1['loss'].backward()
                    lr = opts[0].step(epoch)
                    net.zero_grad()

                    actor_ctrs = gpu(data_sub['actor_ctrs'], gpu_id=config['gpu_id'])
                    actor_idcs = gpu(data_sub['actor_idcs'], gpu_id=config['gpu_id'])
                    actors_hidden = gpu(data_sub['actors_hidden'], gpu_id=config['gpu_id'])
                    nodes = gpu(data_sub['nodes'], gpu_id=config['gpu_id'])
                    graph_idcs = gpu(data_sub['graph_idcs'], gpu_id=config['gpu_id'])
                    ego_feat = gpu(data_sub['ego_feat'], gpu_id=config['gpu_id'])
                    feats = gpu(data_sub['feats'], gpu_id=config['gpu_id'])
                    nearest_ctrs_hist = gpu(data_sub['nearest_ctrs_hist'], gpu_id=config['gpu_id'])
                    rot = gpu(data_sub['rot'], gpu_id=config['gpu_id'])
                    orig = gpu(data_sub['orig'], gpu_id=config['gpu_id'])
                    gt_preds = gpu(data_sub['gt_preds'], gpu_id=config['gpu_id'])
                    has_preds = gpu(data_sub['has_preds'], gpu_id=config['gpu_id'])
                    ego_feat_calc = gpu(data_sub['ego_feat_calc'], gpu_id=config['gpu_id'])
                    actors = gpu(data_sub['actors'], gpu_id=config['gpu_id'])
                    graph_mod = gpu(data_sub['graph_mod'], gpu_id=config['gpu_id'])

                    inputs = [actor_ctrs, actor_idcs, actors_hidden, nodes, graph_idcs, ego_feat, feats, nearest_ctrs_hist, rot, orig, ego_feat_calc, actors, data, graph_mod]
                    out = net(inputs)
                    out_non_interact = out[0]
                    out_sur_interact = out[1]

                    gt_new = [(torch.repeat_interleave(gt_preds[i].unsqueeze(dim=1), 6, dim=1)-out_non_interact['reg'][i]).detach() for i in range(len(gt_preds))]
                    loss_out2 = losses[1](out_sur_interact, gt_new, has_preds)
                    opts[1].zero_grad()
                    loss_out2.backward()
                    lr = opts[1].step(epoch)

                    out_added = out_non_interact
                    out_added['reg'] = [out_added['reg'][i] + out_sur_interact['reg'][i] for i in range(len(out_added['reg']))]
                    post_out = post_process(out_added, gt_preds, has_preds)
                    ade1, fde1, ade, fde, _ = pred_metrics(np.concatenate(post_out['preds'], 0),
                                                           np.concatenate(post_out['gt_preds'], 0),
                                                           np.concatenate(post_out['has_preds'], 0))

                    ade1_tot += ade1 * len(data["city"])
                    fde1_tot += fde1 * len(data["city"])
                    ade_tot += ade * len(data["city"])
                    fde_tot += fde * len(data["city"])
                    loss_tot += loss_out1["loss"].item() * len(data["city"])
                    loss_inter_tot += loss_out2.item() * len(data["city"])
                    update_num += len(data["city"])
                elif args.subcase == 3:
                    gt_new = [(torch.repeat_interleave(gt_preds[i].unsqueeze(dim=1), 6, dim=1)-out_non_interact['reg'][i]).detach() for i in range(len(gt_preds))]
                    loss_out = losses[0](out_sur_interact, gt_new, has_preds)
                    out_added = out_non_interact
                    out_added['reg'] = [out_added['reg'][i] + out_sur_interact['reg'][i] for i in range(len(out_added['reg']))]

                    post_out = post_process(out_added, gt_preds, has_preds)
                    ade1, fde1, ade, fde, _ = pred_metrics(np.concatenate(post_out['preds'], 0),
                                                           np.concatenate(post_out['gt_preds'], 0),
                                                           np.concatenate(post_out['has_preds'], 0))

                    ade1_tot += ade1 * len(data["city"])
                    fde1_tot += fde1 * len(data["city"])
                    ade_tot += ade * len(data["city"])
                    fde_tot += fde * len(data["city"])
                    loss_tot += loss_out.item() * len(data["city"])
                    update_num += len(data["city"])
                    
                    opts[0].zero_grad()
                    loss_out.backward()
                    lr = opts[0].step(epoch)


        if epoch % val_iters == val_iters - 1:
            if not os.path.exists(config["save_dir"]):
                os.makedirs(config["save_dir"])
            if first_val:
                with open(config["save_dir"] + '/info.pickle', 'wb') as f:
                    pickle.dump([args, config], f, pickle.HIGHEST_PROTOCOL)
                first_val = False
            [loss_val, ade1_val, fde1_val, ade6_val, fde6_val] = val(config, val_loader, net, post_process, epoch)
            writer.add_scalar('loss_val', loss_val, epoch)
            writer.add_scalar('ade_val', ade1_val, epoch)
            writer.add_scalar('fde_val', fde1_val, epoch)
            writer.add_scalar('ade6_val', ade6_val, epoch)
            writer.add_scalar('fde6_val', fde6_val, epoch)
        if epoch % save_iters == save_iters - 1:
            if not os.path.exists(config["save_dir"]):
                os.makedirs(config["save_dir"])
            save_ckpt(net, opts, config["save_dir"], epoch)
        writer.add_scalar('loss_train', loss_tot / update_num, epoch)
        writer.add_scalar('ade_train', ade1_tot / update_num, epoch)
        writer.add_scalar('fde_train', fde1_tot / update_num, epoch)
        writer.add_scalar('ade6_train', ade_tot / update_num, epoch)
        writer.add_scalar('fde6_train', fde_tot / update_num, epoch)


def val(config, data_loader, net, post_process, epoch):
    net.eval()
    update_num = 0
    ade1_tot = 0
    fde1_tot = 0
    ade_tot = 0
    fde_tot = 0
    loss_tot = 0
    batch_num = len(data_loader.dataset)
    init_time = time.time()
    with torch.no_grad():
        for i, data in enumerate(data_loader):
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
            actors_hidden = gpu(data['actors_hidden'], gpu_id=config['gpu_id'])
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
            actors = gpu(data['actors'], gpu_id=config['gpu_id'])
            graph_mod = gpu(data['graph_mod'], gpu_id=config['gpu_id'])

            inputs = [actor_ctrs, actor_idcs, actors_hidden, nodes, graph_idcs, ego_feat, feats, nearest_ctrs_hist, rot, orig, ego_feat_calc, actors, data, graph_mod]
            out = net(inputs)

            if args.case == 0 and args.subcase == 1:
                post_out = post_process(out, gt_preds, has_preds)
                ade1, fde1, ade, fde, _ = pred_metrics(np.concatenate(post_out['preds'], 0),
                                                       np.concatenate(post_out['gt_preds'], 0),
                                                       np.concatenate(post_out['has_preds'], 0))

                ade1_tot += ade1 * len(data["city"])
                fde1_tot += fde1 * len(data["city"])
                ade_tot += ade * len(data["city"])
                fde_tot += fde * len(data["city"])
                update_num += len(data["city"])

            elif args.case == 1:
                out_non_interact = out[0]
                out_sur_interact = out[1]
                out_tot = out[2]
                if args.subcase == 1:
                    post_out = post_process(out_tot, gt_preds, has_preds)
                    ade1, fde1, ade, fde, _ = pred_metrics(np.concatenate(post_out['preds'], 0),
                                                           np.concatenate(post_out['gt_preds'], 0),
                                                           np.concatenate(post_out['has_preds'], 0))

                    ade1_tot += ade1 * len(data["city"])
                    fde1_tot += fde1 * len(data["city"])
                    ade_tot += ade * len(data["city"])
                    fde_tot += fde * len(data["city"])
                    update_num += len(data["city"])

                elif args.subcase == 2:
                    out_added = out_non_interact
                    out_added['reg'] = [out_added['reg'][i] + out_sur_interact['reg'][i] for i in range(len(out_added['reg']))]
                    post_out = post_process(out_added, gt_preds, has_preds)
                    ade1, fde1, ade, fde, _ = pred_metrics(np.concatenate(post_out['preds'], 0),
                                                           np.concatenate(post_out['gt_preds'], 0),
                                                           np.concatenate(post_out['has_preds'], 0))

                    ade1_tot += ade1 * len(data["city"])
                    fde1_tot += fde1 * len(data["city"])
                    ade_tot += ade * len(data["city"])
                    fde_tot += fde * len(data["city"])
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

# TODO: revise map consideration << prediction head 쪽에서 고려하는 방향으로
# TODO: