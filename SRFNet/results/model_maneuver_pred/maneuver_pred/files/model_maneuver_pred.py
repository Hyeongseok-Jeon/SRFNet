# Copyright (c) 2020 Uber Technologies, Inc.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import os
import sys
from fractions import gcd
from numbers import Number

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from data import ArgoDataset, collate_fn
from utils import gpu, to_long, Optimizer, StepLR

from layers import Conv1d, Res1d, Linear, LinearRes, Null, GraphAttentionLayer, GraphAttentionLayer_time_serial, GAT_SRF
from numpy import float64, ndarray
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

file_path = os.path.abspath(__file__)
root_path = os.path.dirname(file_path)
# root_path = os.getcwd()
model_name = os.path.basename(file_path).split(".")[0]

### config ###
config = dict()
"""Train"""
config["display_iters"] = 205942
config["val_iters"] = 205942 * 2
config["save_freq"] = 1.0
config["epoch"] = 0
config["horovod"] = True
config["opt"] = "adam"
config["num_epochs"] = 32
config["lr"] = [1e-3, 1e-4]
config["lr_epochs"] = [32]
config["lr_func"] = StepLR(config["lr"], config["lr_epochs"])

if "save_dir" not in config:
    config["save_dir"] = os.path.join(
        root_path, "results", model_name
    )

if not os.path.isabs(config["save_dir"]):
    config["save_dir"] = os.path.join(root_path, "results", config["save_dir"])

config["batch_size"] = 2
config["val_batch_size"] = 2
config["workers"] = 32
config["val_workers"] = config["workers"]

"""Dataset"""
# Raw Dataset
config["train_split"] = os.path.join(
    root_path, "dataset/train/data"
)
config["val_split"] = os.path.join(root_path, "dataset/val/data")
config["test_split"] = os.path.join(root_path, "dataset/test_obs/data")

# Preprocessed Dataset
config["preprocess"] = True  # whether use preprocess or not
config["preprocess_train"] = os.path.join(
    root_path, "dataset", "preprocess", "train_crs_dist6_angle90.p"
)
config["preprocess_val"] = os.path.join(
    root_path, "dataset", "preprocess", "val_crs_dist6_angle90.p"
)
config['preprocess_test'] = os.path.join(root_path, "dataset", 'preprocess', 'test_test.p')
config["training"] = True

"""Model"""
config["rot_aug"] = False
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


### end of config ###

class maneuver_pred_net(nn.Module):
    def __init__(self, config):
        super(maneuver_pred_net, self).__init__()
        self.config = config

        self.man_classify_net = ManNet(config)

    def forward(self, data: Dict) -> Dict[str, List[Tensor]]:
        # construct actor feature

        _, actor_idcs, _ = actor_gather(gpu(data["feats"]))
        cl_cands, cl_idcs = cl_gather(gpu(data["cl_cands_mod"]))
        target_idcs = [actor_idcs[i][1] for i in range(len(actor_idcs))]

        out_raw = self.man_classify_net(cl_cands)
        out = self.out_reform(out_raw, cl_idcs)

        return out, target_idcs

    def out_reform(self, out_raw, cl_idcs):
        veh_num = len(cl_idcs)
        out_mod = []
        for i in range(veh_num):
            out_mod.append(out_raw[cl_idcs[i]][:,0])

        return out_mod


class ManNet(nn.Module):
    def __init__(self, config):
        super(ManNet, self).__init__()
        self.config = config
        convs = []

        channel_list = [4, 8, 16, 32, 64]
        for i in range(len(channel_list)-1):
            conv = nn.Conv1d(in_channels=channel_list[i],
                             out_channels=channel_list[i+1],
                             kernel_size=4,
                             stride=2,
                             padding=0,
                             dilation=1)
            convs.append(conv)
        self.convs = nn.Sequential(*convs).double()

        self.output = nn.Linear(channel_list[-1], 1)

    def forward(self, cl_cands):
        out = self.convs(cl_cands)
        out = torch.squeeze(out)
        out = self.output(out)
        out = torch.sigmoid(out)

        return out


def actor_gather(actors: List[Tensor]) -> Tuple[Tensor, List[Tensor]]:
    batch_size = len(actors)
    num_actors = [len(x) for x in actors]

    actors = [x.transpose(1, 2) for x in actors]
    actors = torch.cat(actors, 0)

    mask = torch.zeros_like(actors)
    actors_inter_cat = []
    for i in range(5):
        if i == 0:
            element = mask.clone()
            element[:, :, -1] = actors[:, :, 0]
            actors_inter_cat.append(element)
        else:
            element = mask.clone()
            element[:, :, -5 * i:] = actors[:, :, :5 * i]
            actors_inter_cat.append(element)

    actor_idcs = []
    count = 0
    for i in range(batch_size):
        idcs = torch.arange(count, count + num_actors[i]).to(actors.device)
        actor_idcs.append(idcs)
        count += num_actors[i]
    return actors, actor_idcs, actors_inter_cat


def cl_gather(cl_cands):
    cl_cands_cat = []
    cl_idcs = []
    veh_calc = 0
    for i in range(len(cl_cands)):
        for j in range(len(cl_cands[i])):
            cl_cands_cat.append(cl_cands[i][j])
            idx = [i + veh_calc for i in range(cl_cands[i][j].shape[0])]
            cl_idcs.append(idx)
            veh_calc = veh_calc + cl_cands[i][j].shape[0]
    cl_cands_cat = torch.cat(cl_cands_cat, dim=0)
    cl_cands_cat = torch.transpose(cl_cands_cat, 1, 2)

    return cl_cands_cat, cl_idcs


class Loss(nn.Module):
    def __init__(self, config):
        super(Loss, self).__init__()
        self.config = config
        self.class_loss = nn.CrossEntropyLoss()

    def forward(self, out, gt_mod):
        loss_calc_num = 0
        loss = 0
        val_idx = []
        pred_idx = []
        for i in range(len(out)):
            if not (len(gt_mod[i]) == 1 and gt_mod[i][0] == 0):
                pred_idx.append(i)
                if len(gt_mod[i]) > 1:
                    pred = out[i].unsqueeze(dim=0)
                    gt = gpu(torch.where(gt_mod[i]==1)[0])

                    self.class_loss(pred, gt)
                    loss_calc_num += 1
                    loss += self.class_loss(pred, gt)
                    val_idx.append(i)
        return loss, loss_calc_num, val_idx, pred_idx


class PostProcess(nn.Module):
    def __init__(self, config):
        super(PostProcess, self).__init__()
        self.config = config

    def forward(self, out, target_idcs, data):
        post_out = dict()
        post_out["out"] = [out[i].detach().cpu().numpy() for i in target_idcs]
        post_out["gt_preds"] = [data["gt_cl_cands"][i][1] for i in range(len(target_idcs))]
        return post_out

    def append(self, metrics, loss, loss_calc_num, post_out):
        if len(metrics.keys()) == 0:
            metrics['loss'] = 0.0
            metrics['calc_num'] = 0
            for key in post_out:
                metrics[key] = []

        if isinstance(loss, torch.Tensor):
            metrics['loss'] += loss.item()
            metrics['calc_num'] += loss_calc_num
        else:
            metrics['loss'] += loss
            metrics['calc_num'] += loss_calc_num

        for key in post_out:
            metrics[key] += post_out[key]
        return metrics

    def display(self, metrics, dt, epoch, lr=None):
        """Every display-iters print training/val information"""
        if lr is not None:
            print("Epoch %3.3f, lr %.5f, time %3.2f" % (epoch, lr, dt))
        else:
            print(
                "************************* Validation, time %3.2f *************************"
                % dt
            )

        loss = metrics["loss"]
        calc_num = metrics['calc_num']

        preds = metrics["out"]
        gt_preds = metrics["gt_preds"]

        acc = pred_metrics(preds, gt_preds)

        print(
            "loss %2.4f, accuracy %2.4f %%"
            % (loss/calc_num, acc)
        )
        print()


def pred_metrics(preds, gt_preds):
    tot_num = len(preds)
    correct_num = 0
    for i in range(tot_num):
        pred = np.argmax(preds[i])
        gt = np.where(gt_preds[i]==1)[0][0]
        if pred == gt:
            correct_num += 1
    return correct_num*100 / tot_num


def get_model(args):
    net = maneuver_pred_net(config).double().cuda()
    params = net.parameters()
    opt = Optimizer(params, config)
    loss = Loss(config).cuda()
    post_process = PostProcess(config).cuda()

    config["save_dir"] = os.path.join(
        config["save_dir"], args.case
    )

    return config, ArgoDataset, collate_fn, net, loss, post_process, opt
