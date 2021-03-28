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

class lanegcn(nn.Module):
    def __init__(self, config, args):
        super(lanegcn, self).__init__()
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
        actors, actor_idcs, _ = actor_gather(gpu(data["feats"]))
        actor_ctrs = gpu(data["ctrs"])
        '''
        actors : N x 3 x 20 (N : number of vehicles in every batches)
        '''

        actors = self.actor_net(actors)

        # construct map features
        graph = graph_gather(to_long(gpu(data["graph"])))
        '''
        graph['idcs'] : list with length or batch size, graph['idcs'][i]
        '''
        nodes, node_idcs, node_ctrs = self.map_net(graph)

        # actor-map fusion cycle 
        nodes = self.a2m(nodes, graph, actors, actor_idcs, actor_ctrs)
        nodes = self.m2m(nodes, graph)
        actors = self.m2a(actors, actor_idcs, actor_ctrs, nodes, node_idcs, node_ctrs)
        actors = self.a2a(actors, actor_idcs, actor_ctrs)

        # prediction
        '''
        actors : N x 128 (N : number of vehicles in every batches)
        '''
        out = self.pred_net(actors, actor_idcs, actor_ctrs)
        rot, orig = gpu(data["rot"]), gpu(data["orig"])
        # transform prediction to world coordinates
        for i in range(len(out["reg"])):
            out["reg"][i] = torch.matmul(out["reg"][i], rot[i]) + orig[i].view(
                1, 1, 1, -1
            )
        return [out]


class case_1_1(nn.Module):
    def __init__(self, config, args):
        super(case_1_1, self).__init__()
        self.config = config

        self.actor_net = ActorNet(config)
        self.pred_net_tnt = PredNet_tnt(config)

    def forward(self, data: Dict) -> Dict[str, List[Tensor]]:
        # construct actor feature
        actors, actor_idcs, _ = actor_gather(gpu(data["feats"]))
        actor_ctrs = gpu(data["ctrs"])
        actors = self.actor_net(actors)

        # prediction
        out = self.pred_net(actors, actor_idcs, actor_ctrs)
        rot, orig = gpu(data["rot"]), gpu(data["orig"])
        # transform prediction to world coordinates
        for i in range(len(out["reg"])):
            out["reg"][i] = torch.matmul(out["reg"][i], rot[i]) + orig[i].view(
                1, 1, 1, -1
            )
        return [out]


class case_2_1(nn.Module):
    def __init__(self, config, args):
        super(case_2_1, self).__init__()
        self.config = config

        self.actor_net = ActorNet(config)
        self.map_net = MapNet(config)

        self.fusion_net = FusionNet(config)
        self.inter_pred_net = PredNet(config)
        self.pred_net_tnt = PredNet_tnt(config)

    def forward(self, data: Dict) -> Dict[str, List[Tensor]]:
        # construct actor feature
        actors, actor_idcs, actors_inter_cat = actor_gather(gpu(data["feats"]))
        actor_ctrs = gpu(data["ctrs"])
        actors = self.actor_net(actors)
        actors_inter_cat = self.actor_net(torch.cat(actors_inter_cat))
        actors_inter_cat = torch.cat([actors_inter_cat[actors.shape[0] * i:actors.shape[0] * (i + 1), :].unsqueeze(dim=1) for i in range(5)], dim=1)

        # construct map features
        graph = graph_gather(to_long(gpu(data["graph"])))
        nodes, node_idcs, node_ctrs = self.map_net(graph)

        nearest_ctrs_hist = data['nearest_ctrs_hist']
        for i in range(len(nearest_ctrs_hist)):
            if i == 0:
                nearest_ctrs_hist[i] = nearest_ctrs_hist[i]
            else:
                nearest_ctrs_hist[i] = nearest_ctrs_hist[i] + np.sum(np.asarray([len(node_idcs[j]) for j in range(i)]))
        nearest_ctrs_cat = torch.cat(nearest_ctrs_hist, dim=0)

        '''
        actors : N x 128 (N : number of vehicles in every batches)
        '''
        graph_adjs = []
        for i in range(5):
            if i == 0:
                element = nodes[nearest_ctrs_cat[:, i].long()].unsqueeze(dim=1)
                graph_adjs.append(element)
            else:
                element = nodes[nearest_ctrs_cat[:, 5 * i - 1].long()].unsqueeze(dim=1)
                graph_adjs.append(element)
        graph_adjs = torch.cat(graph_adjs, dim=1)

        # actor-map fusion cycle
        '''
        actors_inter_cat : N x 5 x 128 (N : number of vehicles in every batches)
        graph_adjs : N x 5 x 128 (N : number of vehicles in every batches) [node feature of the nearest node]
        '''
        interaction_mod = self.fusion_net(actors_inter_cat, graph_adjs)

        # prediction
        rot, orig = gpu(data["rot"]), gpu(data["orig"])
        out_non_interact = self.pred_net(actors, actor_idcs, actor_ctrs)
        # transform prediction to world coordinates
        for i in range(len(out_non_interact["reg"])):
            out_non_interact["reg"][i] = torch.matmul(out_non_interact["reg"][i], rot[i]) + orig[i].view(
                1, 1, 1, -1
            )

        out_sur_interact = self.inter_pred_net(interaction_mod, actor_idcs, actor_ctrs)
        for i in range(len(out_sur_interact["reg"])):
            out_sur_interact["reg"][i] = torch.matmul(out_sur_interact["reg"][i], rot[i]) + torch.zeros_like(orig[i]).view(
                1, 1, 1, -1
            )
        for i in range(len(out_sur_interact['reg'])):
            out_sur_interact['reg'][i] = 3 * out_sur_interact['reg'][i]

        out = dict()
        out['cls'] = out_non_interact['cls']
        out['reg'] = [out_non_interact['reg'][i] + out_sur_interact['reg'][i] for i in range(len(out_non_interact['reg']))]
        return [out]


class case_2_2(nn.Module):
    def __init__(self, config, args):
        super(case_2_2, self).__init__()
        self.config = config

        self.actor_net = ActorNet(config)
        self.map_net = MapNet(config)

        self.fusion_net = FusionNet(config)
        self.inter_pred_net = PredNet(config)
        self.pred_net_tnt = PredNet_tnt(config)

    def forward(self, data: Dict) -> Dict[str, List[Tensor]]:
        # construct actor feature
        actors, actor_idcs, actors_inter_cat = actor_gather(gpu(data["feats"]))
        actor_ctrs = gpu(data["ctrs"])
        actors = self.actor_net(actors)
        actors_inter_cat = self.actor_net(torch.cat(actors_inter_cat))
        actors_inter_cat = torch.cat([actors_inter_cat[actors.shape[0] * i:actors.shape[0] * (i + 1), :].unsqueeze(dim=1) for i in range(5)], dim=1)

        # construct map features
        graph = graph_gather(to_long(gpu(data["graph"])))
        nodes, node_idcs, node_ctrs = self.map_net(graph)

        nearest_ctrs_hist = data['nearest_ctrs_hist']
        for i in range(len(nearest_ctrs_hist)):
            if i == 0:
                nearest_ctrs_hist[i] = nearest_ctrs_hist[i]
            else:
                nearest_ctrs_hist[i] = nearest_ctrs_hist[i] + np.sum(np.asarray([len(node_idcs[j]) for j in range(i)]))
        nearest_ctrs_cat = torch.cat(nearest_ctrs_hist, dim=0)

        '''
        actors : N x 128 (N : number of vehicles in every batches)
        '''
        graph_adjs = []
        for i in range(5):
            if i == 0:
                element = nodes[nearest_ctrs_cat[:, i].long()].unsqueeze(dim=1)
                graph_adjs.append(element)
            else:
                element = nodes[nearest_ctrs_cat[:, 5 * i - 1].long()].unsqueeze(dim=1)
                graph_adjs.append(element)
        graph_adjs = torch.cat(graph_adjs, dim=1)

        # actor-map fusion cycle
        '''
        actors_inter_cat : N x 5 x 128 (N : number of vehicles in every batches)
        graph_adjs : N x 5 x 128 (N : number of vehicles in every batches) [node feature of the nearest node]
        '''
        interaction_mod = self.fusion_net(actors_inter_cat, graph_adjs)

        # prediction
        rot, orig = gpu(data["rot"]), gpu(data["orig"])
        out_non_interact = self.pred_net(actors, actor_idcs, actor_ctrs)
        # transform prediction to world coordinates
        for i in range(len(out_non_interact["reg"])):
            out_non_interact["reg"][i] = torch.matmul(out_non_interact["reg"][i], rot[i]) + orig[i].view(
                1, 1, 1, -1
            )

        out_sur_interact = self.inter_pred_net(interaction_mod, actor_idcs, actor_ctrs)
        for i in range(len(out_sur_interact["reg"])):
            out_sur_interact["reg"][i] = torch.matmul(out_sur_interact["reg"][i], rot[i]) + torch.zeros_like(orig[i]).view(
                1, 1, 1, -1
            )
        for i in range(len(out_sur_interact['reg'])):
            out_sur_interact['reg'][i] = 3 * out_sur_interact['reg'][i]

        return [out_non_interact, out_sur_interact]


class case_2_3(nn.Module):
    def __init__(self, config, args):
        super(case_2_3, self).__init__()
        self.config = config

        self.actor_net = ActorNet(config)
        self.map_net = MapNet(config)

        self.fusion_net = FusionNet(config)
        self.inter_pred_net = PredNet(config)
        self.pred_net_tnt = PredNet_tnt(config)

    def forward(self, data: Dict) -> Dict[str, List[Tensor]]:
        # construct actor feature
        actors, actor_idcs, actors_inter_cat = actor_gather(gpu(data["feats"]))
        actor_ctrs = gpu(data["ctrs"])
        actors = self.actor_net(actors)
        actors_inter_cat = self.actor_net(torch.cat(actors_inter_cat))
        actors_inter_cat = torch.cat([actors_inter_cat[actors.shape[0] * i:actors.shape[0] * (i + 1), :].unsqueeze(dim=1) for i in range(5)], dim=1)

        # construct map features
        graph = graph_gather(to_long(gpu(data["graph"])))
        nodes, node_idcs, node_ctrs = self.map_net(graph)

        nearest_ctrs_hist = data['nearest_ctrs_hist']
        for i in range(len(nearest_ctrs_hist)):
            if i == 0:
                nearest_ctrs_hist[i] = nearest_ctrs_hist[i]
            else:
                nearest_ctrs_hist[i] = nearest_ctrs_hist[i] + np.sum(np.asarray([len(node_idcs[j]) for j in range(i)]))
        nearest_ctrs_cat = torch.cat(nearest_ctrs_hist, dim=0)

        '''
        actors : N x 128 (N : number of vehicles in every batches)
        '''
        graph_adjs = []
        for i in range(5):
            if i == 0:
                element = nodes[nearest_ctrs_cat[:, i].long()].unsqueeze(dim=1)
                graph_adjs.append(element)
            else:
                element = nodes[nearest_ctrs_cat[:, 5 * i - 1].long()].unsqueeze(dim=1)
                graph_adjs.append(element)
        graph_adjs = torch.cat(graph_adjs, dim=1)

        # actor-map fusion cycle
        '''
        actors_inter_cat : N x 5 x 128 (N : number of vehicles in every batches)
        graph_adjs : N x 5 x 128 (N : number of vehicles in every batches) [node feature of the nearest node]
        '''
        interaction_mod = self.fusion_net(actors_inter_cat, graph_adjs)

        # prediction
        rot, orig = gpu(data["rot"]), gpu(data["orig"])
        out_non_interact = self.pred_net(actors, actor_idcs, actor_ctrs)
        # transform prediction to world coordinates
        for i in range(len(out_non_interact["reg"])):
            out_non_interact["reg"][i] = torch.matmul(out_non_interact["reg"][i], rot[i]) + orig[i].view(
                1, 1, 1, -1
            )

        out_sur_interact = self.inter_pred_net(interaction_mod, actor_idcs, actor_ctrs)
        for i in range(len(out_sur_interact["reg"])):
            out_sur_interact["reg"][i] = torch.matmul(out_sur_interact["reg"][i], rot[i]) + torch.zeros_like(orig[i]).view(
                1, 1, 1, -1
            )
        for i in range(len(out_sur_interact['reg'])):
            out_sur_interact['reg'][i] = 3 * out_sur_interact['reg'][i]

        return [out_non_interact, out_sur_interact]


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


def graph_gather(graphs):
    batch_size = len(graphs)
    node_idcs = []
    count = 0
    counts = []
    for i in range(batch_size):
        counts.append(count)
        idcs = torch.arange(count, count + graphs[i]["num_nodes"]).to(
            graphs[i]["feats"].device
        )
        node_idcs.append(idcs)
        count = count + graphs[i]["num_nodes"]

    graph = dict()
    graph["idcs"] = node_idcs
    graph["ctrs"] = [x["ctrs"] for x in graphs]

    for key in ["feats", "turn", "control", "intersect"]:
        graph[key] = torch.cat([x[key] for x in graphs], 0)

    for k1 in ["pre", "suc"]:
        graph[k1] = []
        for i in range(len(graphs[0]["pre"])):
            graph[k1].append(dict())
            for k2 in ["u", "v"]:
                graph[k1][i][k2] = torch.cat(
                    [graphs[j][k1][i][k2] + counts[j] for j in range(batch_size)], 0
                )

    for k1 in ["left", "right"]:
        graph[k1] = dict()
        for k2 in ["u", "v"]:
            temp = [graphs[i][k1][k2] + counts[i] for i in range(batch_size)]
            temp = [
                x if x.dim() > 0 else graph["pre"][0]["u"].new().resize_(0)
                for x in temp
            ]
            graph[k1][k2] = torch.cat(temp)
    return graph


class ActorNet(nn.Module):
    """
    Actor feature extractor with Conv1D
    """

    def __init__(self, config):
        super(ActorNet, self).__init__()
        self.config = config
        norm = "GN"
        ng = 1

        n_in = 3
        n_out = [32, 64, 128]
        blocks = [Res1d, Res1d, Res1d]
        num_blocks = [2, 2, 2]

        groups = []
        for i in range(len(num_blocks)):
            group = []
            if i == 0:
                group.append(blocks[i](n_in, n_out[i], norm=norm, ng=ng))
            else:
                group.append(blocks[i](n_in, n_out[i], stride=2, norm=norm, ng=ng))

            for j in range(1, num_blocks[i]):
                group.append(blocks[i](n_out[i], n_out[i], norm=norm, ng=ng))
            groups.append(nn.Sequential(*group))
            n_in = n_out[i]
        self.groups = nn.ModuleList(groups)

        n = config["n_actor"]
        lateral = []
        for i in range(len(n_out)):
            lateral.append(Conv1d(n_out[i], n, norm=norm, ng=ng, act=False))
        self.lateral = nn.ModuleList(lateral)

        self.output = Res1d(n, n, norm=norm, ng=ng)

    def forward(self, actors: Tensor) -> Tensor:
        out = actors

        outputs = []
        for i in range(len(self.groups)):
            out = self.groups[i](out)
            outputs.append(out)

        out = self.lateral[-1](outputs[-1])
        for i in range(len(outputs) - 2, -1, -1):
            out = F.interpolate(out, scale_factor=2, mode="linear", align_corners=False)
            out += self.lateral[i](outputs[i])

        out = self.output(out)[:, :, -1]
        return out


class MapNet(nn.Module):
    """
    Map Graph feature extractor with LaneGraphCNN
    """

    def __init__(self, config):
        super(MapNet, self).__init__()
        self.config = config
        n_map = config["n_map"]
        norm = "GN"
        ng = 1

        self.input = nn.Sequential(
            nn.Linear(2, n_map),
            nn.ReLU(inplace=True),
            Linear(n_map, n_map, norm=norm, ng=ng, act=False),
        )
        self.seg = nn.Sequential(
            nn.Linear(2, n_map),
            nn.ReLU(inplace=True),
            Linear(n_map, n_map, norm=norm, ng=ng, act=False),
        )

        keys = ["ctr", "norm", "ctr2", "left", "right"]
        for i in range(config["num_scales"]):
            keys.append("pre" + str(i))
            keys.append("suc" + str(i))

        fuse = dict()
        for key in keys:
            fuse[key] = []

        for i in range(4):
            for key in fuse:
                if key in ["norm"]:
                    fuse[key].append(nn.GroupNorm(gcd(ng, n_map), n_map))
                elif key in ["ctr2"]:
                    fuse[key].append(Linear(n_map, n_map, norm=norm, ng=ng, act=False))
                else:
                    fuse[key].append(nn.Linear(n_map, n_map, bias=False))

        for key in fuse:
            fuse[key] = nn.ModuleList(fuse[key])
        self.fuse = nn.ModuleDict(fuse)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, graph):
        if (
                len(graph["feats"]) == 0
                or len(graph["pre"][-1]["u"]) == 0
                or len(graph["suc"][-1]["u"]) == 0
        ):
            temp = graph["feats"]
            return (
                temp.new().resize_(0),
                [temp.new().long().resize_(0) for x in graph["node_idcs"]],
                temp.new().resize_(0),
            )

        ctrs = torch.cat(graph["ctrs"], 0)
        feat = self.input(ctrs)
        feat += self.seg(graph["feats"])
        feat = self.relu(feat)

        """fuse map"""
        res = feat
        for i in range(len(self.fuse["ctr"])):
            temp = self.fuse["ctr"][i](feat)
            for key in self.fuse:
                if key.startswith("pre") or key.startswith("suc"):
                    k1 = key[:3]
                    k2 = int(key[3:])
                    temp.index_add_(
                        0,
                        graph[k1][k2]["u"],
                        self.fuse[key][i](feat[graph[k1][k2]["v"]]),
                    )

            if len(graph["left"]["u"] > 0):
                temp.index_add_(
                    0,
                    graph["left"]["u"],
                    self.fuse["left"][i](feat[graph["left"]["v"]]),
                )
            if len(graph["right"]["u"] > 0):
                temp.index_add_(
                    0,
                    graph["right"]["u"],
                    self.fuse["right"][i](feat[graph["right"]["v"]]),
                )

            feat = self.fuse["norm"][i](temp)
            feat = self.relu(feat)

            feat = self.fuse["ctr2"][i](feat)
            feat += res
            feat = self.relu(feat)
            res = feat
        return feat, graph["idcs"], graph["ctrs"]


class FusionNet(nn.Module):
    def __init__(self, config):
        super(FusionNet, self).__init__()
        self.config = config
        self.seq_len = 20
        self.GAT_lstm = GAT(config)

        self.h0 = nn.Parameter(torch.empty(size=(1, config["n_actor"])))
        nn.init.xavier_uniform_(self.h0.data, gain=1.414)

        conv1ds = []
        for i in range(3):
            conv1ds.append(nn.Conv1d(in_channels=128,
                                     out_channels=128,
                                     kernel_size=2,
                                     stride=1,
                                     padding=0,
                                     dilation=1,
                                     groups=1,
                                     bias=True,
                                     padding_mode='zeros'))

        self.conv1d = nn.Sequential(*conv1ds)

    def forward(self, actors_inter_cat, graph_adjs):
        c0 = actors_inter_cat[:, 0, :]
        h0 = torch.repeat_interleave(self.h0, actors_inter_cat.shape[0], dim=0)
        out = []
        for i in range(int(4)):
            out_tmp, [c0, h0] = self.GAT_lstm(actors_inter_cat[:, i + 1, :], graph_adjs[:, i, :], [c0, h0])
            out.append(F.sigmoid(out_tmp))
        out = torch.cat([out[i].unsqueeze(dim=2) for i in range(len(out))], dim=2)
        out = self.conv1d(out).squeeze()
        return out


class A2M(nn.Module):
    """
    Actor to Map Fusion:  fuses real-time traffic information from
    actor nodes to lane nodes
    """

    def __init__(self, config):
        super(A2M, self).__init__()
        self.config = config
        n_map = config["n_map"]
        norm = "GN"
        ng = 1

        """fuse meta, static, dyn"""
        self.meta = Linear(n_map + 4, n_map, norm=norm, ng=ng)
        att = []
        for i in range(2):
            att.append(Att(n_map, config["n_actor"]))
        self.att = nn.ModuleList(att)

    def forward(self, feat: Tensor, graph: Dict[str, Union[List[Tensor], Tensor, List[Dict[str, Tensor]], Dict[str, Tensor]]], actors: Tensor, actor_idcs: List[Tensor], actor_ctrs: List[Tensor]) -> Tensor:
        """meta, static and dyn fuse using attention"""
        meta = torch.cat(
            (
                graph["turn"],
                graph["control"].unsqueeze(1),
                graph["intersect"].unsqueeze(1),
            ),
            1,
        )
        feat = self.meta(torch.cat((feat, meta), 1))

        for i in range(len(self.att)):
            feat = self.att[i](
                feat,
                graph["idcs"],
                graph["ctrs"],
                actors,
                actor_idcs,
                actor_ctrs,
                self.config["actor2map_dist"],
            )
        return feat


class M2M(nn.Module):
    """
    The lane to lane block: propagates information over lane
            graphs and updates the features of lane nodes
    """

    def __init__(self, config):
        super(M2M, self).__init__()
        self.config = config
        n_map = config["n_map"]
        norm = "GN"
        ng = 1

        keys = ["ctr", "norm", "ctr2", "left", "right"]
        for i in range(config["num_scales"]):
            keys.append("pre" + str(i))
            keys.append("suc" + str(i))

        fuse = dict()
        for key in keys:
            fuse[key] = []

        for i in range(4):
            for key in fuse:
                if key in ["norm"]:
                    fuse[key].append(nn.GroupNorm(gcd(ng, n_map), n_map))
                elif key in ["ctr2"]:
                    fuse[key].append(Linear(n_map, n_map, norm=norm, ng=ng, act=False))
                else:
                    fuse[key].append(nn.Linear(n_map, n_map, bias=False))

        for key in fuse:
            fuse[key] = nn.ModuleList(fuse[key])
        self.fuse = nn.ModuleDict(fuse)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, feat: Tensor, graph: Dict) -> Tensor:
        """fuse map"""
        res = feat
        for i in range(len(self.fuse["ctr"])):
            temp = self.fuse["ctr"][i](feat)
            for key in self.fuse:
                if key.startswith("pre") or key.startswith("suc"):
                    k1 = key[:3]
                    k2 = int(key[3:])
                    temp.index_add_(
                        0,
                        graph[k1][k2]["u"],
                        self.fuse[key][i](feat[graph[k1][k2]["v"]]),
                    )

            if len(graph["left"]["u"] > 0):
                temp.index_add_(
                    0,
                    graph["left"]["u"],
                    self.fuse["left"][i](feat[graph["left"]["v"]]),
                )
            if len(graph["right"]["u"] > 0):
                temp.index_add_(
                    0,
                    graph["right"]["u"],
                    self.fuse["right"][i](feat[graph["right"]["v"]]),
                )

            feat = self.fuse["norm"][i](temp)
            feat = self.relu(feat)

            feat = self.fuse["ctr2"][i](feat)
            feat += res
            feat = self.relu(feat)
            res = feat
        return feat


class M2A(nn.Module):
    """
    The lane to actor block fuses updated
        map information from lane nodes to actor nodes
    """

    def __init__(self, config):
        super(M2A, self).__init__()
        self.config = config
        norm = "GN"
        ng = 1

        n_actor = config["n_actor"]
        n_map = config["n_map"]

        att = []
        for i in range(2):
            att.append(Att(n_actor, n_map))
        self.att = nn.ModuleList(att)

    def forward(self, actors: Tensor, actor_idcs: List[Tensor], actor_ctrs: List[Tensor], nodes: Tensor, node_idcs: List[Tensor], node_ctrs: List[Tensor]) -> Tensor:
        for i in range(len(self.att)):
            actors = self.att[i](
                actors,
                actor_idcs,
                actor_ctrs,
                nodes,
                node_idcs,
                node_ctrs,
                self.config["map2actor_dist"],
            )
        return actors


class A2A(nn.Module):
    """
    The actor to actor block performs interactions among actors.
    """

    def __init__(self, config):
        super(A2A, self).__init__()
        self.config = config
        norm = "GN"
        ng = 1

        n_actor = config["n_actor"]
        n_map = config["n_map"]

        att = []
        for i in range(2):
            att.append(Att(n_actor, n_actor))
        self.att = nn.ModuleList(att)

    def forward(self, actors: Tensor, actor_idcs: List[Tensor], actor_ctrs: List[Tensor]) -> Tensor:
        for i in range(len(self.att)):
            actors = self.att[i](
                actors,
                actor_idcs,
                actor_ctrs,
                actors,
                actor_idcs,
                actor_ctrs,
                self.config["actor2actor_dist"],
            )
        return actors


class EncodeDist(nn.Module):
    def __init__(self, n, linear=True):
        super(EncodeDist, self).__init__()
        norm = "GN"
        ng = 1

        block = [nn.Linear(2, n), nn.ReLU(inplace=True)]

        if linear:
            block.append(nn.Linear(n, n))

        self.block = nn.Sequential(*block)

    def forward(self, dist):
        x, y = dist[:, :1], dist[:, 1:]
        dist = torch.cat(
            (
                torch.sign(x) * torch.log(torch.abs(x) + 1.0),
                torch.sign(y) * torch.log(torch.abs(y) + 1.0),
            ),
            1,
        )

        dist = self.block(dist)
        return dist


class PredNet(nn.Module):
    """
    Final motion forecasting with Linear Residual block
    """

    def __init__(self, config):
        super(PredNet, self).__init__()
        self.config = config
        norm = "GN"
        ng = 1

        n_actor = config["n_actor"]

        pred = []
        for i in range(config["num_mods"]):
            pred.append(
                nn.Sequential(
                    LinearRes(n_actor, n_actor, norm=norm, ng=ng),
                    nn.Linear(n_actor, 2 * config["num_preds"]),
                )
            )
        self.pred = nn.ModuleList(pred)

        self.att_dest = AttDest(n_actor)
        self.cls = nn.Sequential(
            LinearRes(n_actor, n_actor, norm=norm, ng=ng), nn.Linear(n_actor, 1)
        )

    def forward(self, actors: Tensor, actor_idcs: List[Tensor], actor_ctrs: List[Tensor]) -> Dict[str, List[Tensor]]:
        preds = []
        for i in range(len(self.pred)):
            preds.append(self.pred[i](actors))
        reg = torch.cat([x.unsqueeze(1) for x in preds], 1)
        reg = reg.view(reg.size(0), reg.size(1), -1, 2)

        for i in range(len(actor_idcs)):
            idcs = actor_idcs[i]
            ctrs = actor_ctrs[i].view(-1, 1, 1, 2)
            reg[idcs] = reg[idcs] + ctrs

        dest_ctrs = reg[:, :, -1].detach()
        feats = self.att_dest(actors, torch.cat(actor_ctrs, 0), dest_ctrs)
        cls = self.cls(feats).view(-1, self.config["num_mods"])

        cls, sort_idcs = cls.sort(1, descending=True)
        row_idcs = torch.arange(len(sort_idcs)).long().to(sort_idcs.device)
        row_idcs = row_idcs.view(-1, 1).repeat(1, sort_idcs.size(1)).view(-1)
        sort_idcs = sort_idcs.view(-1)
        reg = reg[row_idcs, sort_idcs].view(cls.size(0), cls.size(1), -1, 2)

        out = dict()
        out["cls"], out["reg"] = [], []
        for i in range(len(actor_idcs)):
            idcs = actor_idcs[i]
            ctrs = actor_ctrs[i].view(-1, 1, 1, 2)
            out["cls"].append(cls[idcs])
            out["reg"].append(reg[idcs])
        return out


class PredNet_tnt(nn.Module):
    """
    Final motion forecasting with Linear Residual block
    """

    def __init__(self, config):
        super(PredNet_tnt, self).__init__()
        self.config = config
        norm = "GN"
        ng = 1

        n_actor = config["n_actor"]

        pred = []
        for i in range(config["num_mods"]):
            pred.append(
                nn.Sequential(
                    LinearRes(n_actor, n_actor, norm=norm, ng=ng),
                    nn.Linear(n_actor, 2 * config["num_preds"]),
                )
            )
        self.pred = nn.ModuleList(pred)

        self.att_dest = AttDest(n_actor)
        self.cls = nn.Sequential(
            LinearRes(n_actor, n_actor, norm=norm, ng=ng), nn.Linear(n_actor, 1)
        )

    def forward(self, actors: Tensor, actor_idcs: List[Tensor], actor_ctrs: List[Tensor]) -> Dict[str, List[Tensor]]:
        preds = []
        for i in range(len(self.pred)):
            preds.append(self.pred[i](actors))
        reg = torch.cat([x.unsqueeze(1) for x in preds], 1)
        reg = reg.view(reg.size(0), reg.size(1), -1, 2)

        for i in range(len(actor_idcs)):
            idcs = actor_idcs[i]
            ctrs = actor_ctrs[i].view(-1, 1, 1, 2)
            reg[idcs] = reg[idcs] + ctrs

        dest_ctrs = reg[:, :, -1].detach()
        feats = self.att_dest(actors, torch.cat(actor_ctrs, 0), dest_ctrs)
        cls = self.cls(feats).view(-1, self.config["num_mods"])

        cls, sort_idcs = cls.sort(1, descending=True)
        row_idcs = torch.arange(len(sort_idcs)).long().to(sort_idcs.device)
        row_idcs = row_idcs.view(-1, 1).repeat(1, sort_idcs.size(1)).view(-1)
        sort_idcs = sort_idcs.view(-1)
        reg = reg[row_idcs, sort_idcs].view(cls.size(0), cls.size(1), -1, 2)

        out = dict()
        out["cls"], out["reg"] = [], []
        for i in range(len(actor_idcs)):
            idcs = actor_idcs[i]
            ctrs = actor_ctrs[i].view(-1, 1, 1, 2)
            out["cls"].append(cls[idcs])
            out["reg"].append(reg[idcs])
        return out


class Att(nn.Module):
    """
    Attention block to pass context nodes information to target nodes
    This is used in Actor2Map, Actor2Actor, Map2Actor and Map2Map
    """

    def __init__(self, n_agt: int, n_ctx: int) -> None:
        super(Att, self).__init__()
        norm = "GN"
        ng = 1

        self.dist = nn.Sequential(
            nn.Linear(2, n_ctx),
            nn.ReLU(inplace=True),
            Linear(n_ctx, n_ctx, norm=norm, ng=ng),
        )

        self.query = Linear(n_agt, n_ctx, norm=norm, ng=ng)

        self.ctx = nn.Sequential(
            Linear(3 * n_ctx, n_agt, norm=norm, ng=ng),
            nn.Linear(n_agt, n_agt, bias=False),
        )

        self.agt = nn.Linear(n_agt, n_agt, bias=False)
        self.norm = nn.GroupNorm(gcd(ng, n_agt), n_agt)
        self.linear = Linear(n_agt, n_agt, norm=norm, ng=ng, act=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, agts: Tensor, agt_idcs: List[Tensor], agt_ctrs: List[Tensor], ctx: Tensor, ctx_idcs: List[Tensor], ctx_ctrs: List[Tensor], dist_th: float) -> Tensor:
        res = agts
        if len(ctx) == 0:
            agts = self.agt(agts)
            agts = self.relu(agts)
            agts = self.linear(agts)
            agts += res
            agts = self.relu(agts)
            return agts

        batch_size = len(agt_idcs)
        hi, wi = [], []
        hi_count, wi_count = 0, 0
        for i in range(batch_size):
            dist = agt_ctrs[i].view(-1, 1, 2) - ctx_ctrs[i].view(1, -1, 2)
            dist = torch.sqrt((dist ** 2).sum(2))
            mask = dist <= dist_th

            idcs = torch.nonzero(mask, as_tuple=False)
            if len(idcs) == 0:
                continue

            hi.append(idcs[:, 0] + hi_count)
            wi.append(idcs[:, 1] + wi_count)
            hi_count += len(agt_idcs[i])
            wi_count += len(ctx_idcs[i])
        hi = torch.cat(hi, 0)
        wi = torch.cat(wi, 0)

        agt_ctrs = torch.cat(agt_ctrs, 0)
        ctx_ctrs = torch.cat(ctx_ctrs, 0)
        dist = agt_ctrs[hi] - ctx_ctrs[wi]
        dist = self.dist(dist)

        query = self.query(agts[hi])

        ctx = ctx[wi]
        ctx = torch.cat((dist, query, ctx), 1)
        ctx = self.ctx(ctx)

        agts = self.agt(agts)
        agts.index_add_(0, hi, ctx)
        agts = self.norm(agts)
        agts = self.relu(agts)

        agts = self.linear(agts)
        agts += res
        agts = self.relu(agts)
        return agts


class AttDest(nn.Module):
    def __init__(self, n_agt: int):
        super(AttDest, self).__init__()
        norm = "GN"
        ng = 1

        self.dist = nn.Sequential(
            nn.Linear(2, n_agt),
            nn.ReLU(inplace=True),
            Linear(n_agt, n_agt, norm=norm, ng=ng),
        )

        self.agt = Linear(2 * n_agt, n_agt, norm=norm, ng=ng)

    def forward(self, agts: Tensor, agt_ctrs: Tensor, dest_ctrs: Tensor) -> Tensor:
        n_agt = agts.size(1)
        num_mods = dest_ctrs.size(1)

        dist = (agt_ctrs.unsqueeze(1) - dest_ctrs).view(-1, 2)
        dist = self.dist(dist)
        agts = agts.unsqueeze(1).repeat(1, num_mods, 1).view(-1, n_agt)

        agts = torch.cat((dist, agts), 1)
        agts = self.agt(agts)
        return agts


class GAT(nn.Module):
    def __init__(self, config):
        super(GAT, self).__init__()

        self.config = config
        self.forget = GAT_SRF(in_features=config['n_actor'] + config['n_map'],
                              out_features=config['n_actor'],
                              dropout=config['GAT_dropout'],
                              alpha=config['GAT_Leakyrelu_alpha'],
                              training=config['training'],
                              concat=True)
        self.input = GAT_SRF(in_features=config['n_actor'] + config['n_map'],
                             out_features=config['n_actor'],
                             dropout=config['GAT_dropout'],
                             alpha=config['GAT_Leakyrelu_alpha'],
                             training=config['training'],
                             concat=True)
        self.input_cell = GAT_SRF(in_features=config['n_actor'] + config['n_map'],
                                  out_features=config['n_actor'],
                                  dropout=config['GAT_dropout'],
                                  alpha=config['GAT_Leakyrelu_alpha'],
                                  training=config['training'],
                                  concat=True)
        self.output = GAT_SRF(in_features=config['n_actor'] + config['n_map'],
                              out_features=config['n_actor'],
                              dropout=config['GAT_dropout'],
                              alpha=config['GAT_Leakyrelu_alpha'],
                              training=config['training'],
                              concat=True)
        self.W = nn.Parameter(torch.empty(size=(config["n_actor"], config["n_actor"])))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.b = nn.Parameter(torch.empty(size=(1, config["n_actor"])))
        nn.init.xavier_uniform_(self.b.data, gain=1.414)

    def forward(self, X, E, feedback):
        '''
        X : (N, config[n_actor])
        E : (N, config[n_map])
        '''
        cell = feedback[0]
        hidden = feedback[1]
        GAT_in = torch.cat((X, hidden), dim=1)
        forget = F.sigmoid(self.forget([GAT_in, E])[0])
        input_gate = F.sigmoid(self.input([GAT_in, E])[0])
        input_cell = F.tanh(self.input_cell([GAT_in, E])[0])
        input = input_gate * input_cell
        output = F.sigmoid(self.output([GAT_in, E])[0])

        cell_new = cell * forget + input
        hidden_new = F.tanh(cell_new) * output
        out = torch.mm(hidden_new, self.W) + torch.repeat_interleave(self.b, X.shape[0], dim=0)
        return out, [cell_new, hidden_new]


class PredLoss(nn.Module):
    def __init__(self, config):
        super(PredLoss, self).__init__()
        self.config = config
        self.reg_loss = nn.SmoothL1Loss(reduction="sum")

    def forward(self, out: Dict[str, List[Tensor]], gt_preds: List[Tensor], has_preds: List[Tensor]) -> Dict[str, Union[Tensor, int]]:
        cls, reg = out["cls"], out["reg"]
        cls = torch.cat([x for x in cls], 0)
        reg = torch.cat([x for x in reg], 0)
        gt_preds = torch.cat([x for x in gt_preds], 0)
        has_preds = torch.cat([x for x in has_preds], 0)

        loss_out = dict()
        zero = 0.0 * (cls.sum() + reg.sum())
        loss_out["cls_loss"] = zero.clone()
        loss_out["num_cls"] = 0
        loss_out["reg_loss"] = zero.clone()
        loss_out["num_reg"] = 0

        num_mods, num_preds = self.config["num_mods"], self.config["num_preds"]
        # assert(has_preds.all())

        last = has_preds.float() + 0.1 * torch.arange(num_preds).float().to(
            has_preds.device
        ) / float(num_preds)
        max_last, last_idcs = last.max(1)
        mask = max_last > 1.0

        cls = cls[mask]
        reg = reg[mask]
        gt_preds = gt_preds[mask]
        has_preds = has_preds[mask]
        last_idcs = last_idcs[mask]

        row_idcs = torch.arange(len(last_idcs)).long().to(last_idcs.device)
        dist = []
        for j in range(num_mods):
            dist.append(
                torch.sqrt(
                    (
                            (reg[row_idcs, j, last_idcs] - gt_preds[row_idcs, last_idcs])
                            ** 2
                    ).sum(1)
                )
            )
        dist = torch.cat([x.unsqueeze(1) for x in dist], 1)
        min_dist, min_idcs = dist.min(1)
        row_idcs = torch.arange(len(min_idcs)).long().to(min_idcs.device)

        mgn = cls[row_idcs, min_idcs].unsqueeze(1) - cls
        mask0 = (min_dist < self.config["cls_th"]).view(-1, 1)
        mask1 = dist - min_dist.view(-1, 1) > self.config["cls_ignore"]
        mgn = mgn[mask0 * mask1]
        mask = mgn < self.config["mgn"]
        coef = self.config["cls_coef"]
        loss_out["cls_loss"] += coef * (
                self.config["mgn"] * mask.sum() - mgn[mask].sum()
        )
        loss_out["num_cls"] += mask.sum().item()

        reg = reg[row_idcs, min_idcs]
        coef = self.config["reg_coef"]
        loss_out["reg_loss"] += coef * self.reg_loss(
            reg[has_preds], gt_preds[has_preds]
        )
        loss_out["num_reg"] += has_preds.sum().item()
        return loss_out


class Loss(nn.Module):
    def __init__(self, config):
        super(Loss, self).__init__()
        self.config = config
        self.pred_loss = PredLoss(config)

    def forward(self, out: Dict, data: Dict) -> Dict:
        loss_out = self.pred_loss(out, gpu(data["gt_preds"]), gpu(data["has_preds"]))
        loss_out["loss"] = loss_out["cls_loss"] / (
                loss_out["num_cls"] + 1e-10
        ) + loss_out["reg_loss"] / (loss_out["num_reg"] + 1e-10)
        return loss_out


class inter_loss(nn.Module):
    def __init__(self, config):
        super(inter_loss, self).__init__()
        self.config = config
        self.pred_loss = torch.nn.L1Loss()

    def forward(self, out, gt, has_preds):
        for i in range(len(out['reg'])):
            mask = ~torch.repeat_interleave(has_preds[i].unsqueeze(dim=-1), 2, dim=-1)
            mask = torch.repeat_interleave(mask.unsqueeze(dim=1), 6, dim=1)
            out['reg'][i].squeeze()[mask] = 0
            for j in range(6):
                gt[i][:, j, :, :][~torch.repeat_interleave(has_preds[i].unsqueeze(dim=-1), 2, dim=-1)] = 0

        loss = self.pred_loss(torch.cat(out['reg'], dim=0), torch.cat(gt, dim=0))

        return loss


class L1loss(nn.Module):
    def __init__(self, config):
        super(L1loss, self).__init__()
        self.config = config
        self.inter_loss = inter_loss(config)

    def forward(self, out: Dict, data: Dict, pred_loss) -> Dict:
        loss = dict()
        for key in pred_loss.keys():
            loss[key] = pred_loss[key]
        loss_out = self.inter_loss(out, gpu(data["gt_new"]), gpu(data["has_preds"]))
        loss["loss"] = loss_out
        return loss


class PostProcess(nn.Module):
    def __init__(self, config):
        super(PostProcess, self).__init__()
        self.config = config

    def forward(self, out, data):
        post_out = dict()
        post_out["preds"] = [x[0:1].detach().cpu().numpy() for x in out["reg"]]
        post_out["gt_preds"] = [x[0:1].numpy() for x in data["gt_preds"]]
        post_out["has_preds"] = [x[0:1].numpy() for x in data["has_preds"]]
        return post_out

    def append(self, metrics: Dict, loss_out: Dict, post_out: Optional[Dict[str, List[ndarray]]] = None) -> Dict:
        if len(metrics.keys()) == 0:
            if len(loss_out) == 1:
                for key in loss_out[0]:
                    if key != "loss":
                        metrics[key] = [0.0]

                for key in post_out:
                    metrics[key] = []
            else:
                for key in loss_out[0]:
                    if key != "loss":
                        metrics[key] = [0.0, 0.0]

                for key in post_out:
                    metrics[key] = []

        for key in loss_out[0]:
            for i in range(len(loss_out)):
                if key == "loss":
                    continue
                if isinstance(loss_out[i][key], torch.Tensor):
                    metrics[key][i] += loss_out[i][key].item()
                else:
                    metrics[key][i] += loss_out[i][key]

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
        loss = []
        for i in range(len(metrics["cls_loss"])):
            cls = metrics["cls_loss"][i] / (metrics["num_cls"][i] + 1e-10)
            reg = metrics["reg_loss"][i] / (metrics["num_reg"][i] + 1e-10)
            loss.append(cls + reg)

        preds = np.concatenate(metrics["preds"], 0)
        gt_preds = np.concatenate(metrics["gt_preds"], 0)
        has_preds = np.concatenate(metrics["has_preds"], 0)
        ade1, fde1, ade, fde, min_idcs = pred_metrics(preds, gt_preds, has_preds)

        if len(loss) == 1:
            print(
                "loss %2.4f %2.4f %2.4f, ade1 %2.4f, fde1 %2.4f, ade %2.4f, fde %2.4f"
                % (loss, cls, reg, ade1, fde1, ade, fde)
            )
            print()
        elif len(loss) == 2:
            print(
                "loss %2.4f %2.4f %2.4f %2.4f, ade1 %2.4f, fde1 %2.4f, ade %2.4f, fde %2.4f"
                % (loss[0], loss[1], cls, reg, ade1, fde1, ade, fde)
            )
            print()


def pred_metrics(preds, gt_preds, has_preds):
    assert has_preds.all()
    preds = np.asarray(preds, np.float32)
    gt_preds = np.asarray(gt_preds, np.float32)

    """batch_size x num_mods x num_preds"""
    err = np.sqrt(((preds - np.expand_dims(gt_preds, 1)) ** 2).sum(3))

    ade1 = err[:, 0].mean()
    fde1 = err[:, 0, -1].mean()

    min_idcs = err[:, :, -1].argmin(1)
    row_idcs = np.arange(len(min_idcs)).astype(np.int64)
    err = err[row_idcs, min_idcs]
    ade = err.mean()
    fde = err[:, -1].mean()
    return ade1, fde1, ade, fde, min_idcs


def get_model(args):
    if args.case == 'case_1_1':
        net = case_1_1(config, args)
        params = net.parameters()
        opt = [Optimizer(params, config)]
        loss = [Loss(config).cuda()]
    elif args.case == 'case_2_1':
        net = case_2_1(config, args)
        params = net.parameters()
        opt = [Optimizer(params, config)]
        loss = [Loss(config).cuda()]
    elif args.case == 'case_2_2':
        net = case_2_2(config, args)
        params1 = list(net.actor_net.parameters()) + list(net.pred_net.parameters())
        params2 = list(net.map_net.parameters()) + list(net.fusion_net.parameters()) + list(net.inter_pred_net.parameters())
        opt = [Optimizer(params1, config), Optimizer(params2, config)]
        loss = [Loss(config).cuda(), L1loss(config).cuda()]
    elif args.case == 'case_2_3':
        net = case_2_3(config, args)
        params2 = list(net.map_net.parameters()) + list(net.fusion_net.parameters()) + list(net.inter_pred_net.parameters())
        opt = [None, Optimizer(params2, config)]
        loss = [Loss(config).cuda(), L1loss(config).cuda()]
    else:
        print('model is not specified. therefore the lanegcn is loaded')
        net = lanegcn(config, args)
        params = net.parameters()
        opt = [Optimizer(params, config)]
        loss = [Loss(config).cuda()]
    net = net.cuda()
    post_process = PostProcess(config).cuda()

    config["save_dir"] = os.path.join(
        config["save_dir"], args.case
    )

    return config, ArgoDataset, collate_fn, net, loss, post_process, opt
