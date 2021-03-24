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

from SRFNet.data import ArgoDataset, collate_fn
from SRFNet.utils import gpu, to_long, Optimizer, StepLR

from SRFNet.layers import Conv1d, Res1d, Linear, LinearRes, Null, GraphAttentionLayer, GraphAttentionLayer_time_serial, GAT_SRF
from numpy import float64, ndarray
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

file_path = os.path.abspath(__file__)
# root_path = os.path.dirname(file_path)
root_path = os.getcwd()
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
config["num_epochs"] = 50
config["lr"] = [1e-3, 1e-4]
config["lr_epochs"] = [32]
config["lr_func"] = StepLR(config["lr"], config["lr_epochs"])

if "save_dir" not in config:
    config["save_dir"] = os.path.join(
        root_path, "results", model_name
    )

if not os.path.isabs(config["save_dir"]):
    config["save_dir"] = os.path.join(root_path, "results", config["save_dir"])

config["batch_size"] = 32
config["val_batch_size"] = 32
config["workers"] = 0
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
    root_path, "SRFNet", "dataset", "preprocess", "train_crs_dist6_angle90.p"
)
config["preprocess_val"] = os.path.join(
    root_path, "SRFNet","dataset", "preprocess", "val_crs_dist6_angle90.p"
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
    def __init__(self, config, args):
        super(maneuver_pred_net, self).__init__()
        self.config = config

        self.actor_net = ActorNet(config)
        self.map_net = MapNet(config)

        self.a2m = A2M(config)
        self.m2m = M2M(config)
        self.m2a = M2A(config)
        self.a2a = A2A(config)
        self.pred_net_tnt = PredNet_tnt(config)

    def forward(self, data: Dict) -> Dict[str, List[Tensor]]:
        # construct actor feature
        cl_cands = cl_cands_gather(gpu(data["cl_cands"]))
        actors, actor_idcs, actors_inter_cat = actor_gather(gpu(data["feats"]))


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


def cl_cands_gather(cl_cands):




def get_model(args):
    net = maneuver_pred_net(config, args)
    params = net.parameters()
    opt = Optimizer(params, config)
    loss = Loss(config).cuda()
    net = net.cuda()
    post_process = PostProcess(config).cuda()

    config["save_dir"] = os.path.join(
        config["save_dir"], args.case
    )

    return config, ArgoDataset, collate_fn, net, loss, post_process, opt
