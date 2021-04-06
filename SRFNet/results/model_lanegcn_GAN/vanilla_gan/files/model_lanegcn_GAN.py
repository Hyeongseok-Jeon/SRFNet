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
from torch.autograd import Variable

from data import ArgoDataset, collate_fn
from utils import gpu, to_long, Optimizer, StepLR, to_float

from layers import Conv1d, Res1d, Linear, LinearRes, Null, GraphAttentionLayer, GraphAttentionLayer_time_serial, GAT_SRF
from model_maneuver_pred import get_model as get_manuever_model
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
    root_path,  "dataset", "preprocess", "train_crs_dist6_angle90.p"
)
config["preprocess_val"] = os.path.join(
    root_path,  "dataset", "preprocess", "val_crs_dist6_angle90.p"
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
config["inter_dist_thres"] = 10
config['gan_noise_dim'] = 128


### end of config ###
class lanegcn_vanilla_gan(nn.Module):
    def __init__(self, config, args):
        super(lanegcn_vanilla_gan, self).__init__()
        self.config = config
        _, _, _, maneuver_pred_net, _, _, _ = get_manuever_model(args)
        pre_trained_weight = torch.load(os.path.join(root_path, "results/model_maneuver_pred/maneuver_pred") + '/32.000.ckpt')
        pretrained_dict = pre_trained_weight['state_dict']
        new_model_dict = maneuver_pred_net.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in new_model_dict}
        new_model_dict.update(pretrained_dict)
        maneuver_pred_net.load_state_dict(new_model_dict)
        self.maneu_pred = maneuver_pred_net

        self.encoder = EncodeNet(config, args)
        self.ego_encoder = EgoEncodeNet(config)
        self.generator = GenerateNet(config)
        self.discriminator = DiscriminateNet(config)

    def forward(self, data):
        actors, actor_idcs, _ = actor_gather(gpu(data["feats"]))
        egos = [actor_idcs[i][0].unsqueeze(dim=0) for i in range(len(actor_idcs))]
        targets = [actor_idcs[i][1].unsqueeze(dim=0) for i in range(len(actor_idcs))]
        actors_targets = [actors[actor_idcs[i][1].unsqueeze(dim=0), :, :] for i in range(len(actor_idcs))]

        rot, orig, ctrs = gpu(data["rot"]), gpu(data["orig"]), [gpu(data["ctrs"])[i][1] for i in range(len(actor_idcs))]
        target_hist_traj = feat_to_global(actors_targets, rot, orig, ctrs)

        hiddens = self.encoder(data)
        hidden_target = hiddens[torch.cat(targets)]
        hidden_ego = hiddens[torch.cat(egos)]

        mu_hidden_ego, log_var_hidden_ego = self.ego_encoder(hidden_ego)
        var_hidden_ego = torch.exp(log_var_hidden_ego * 0.5)

        hidden_noise = Variable(torch.randn(len(hidden_ego), config['gan_noise_dim']).cuda(), requires_grad=True)
        hidden_ego = hidden_noise * var_hidden_ego + mu_hidden_ego

        cl_cands = to_float(gpu(data['cl_cands']))
        actor_ctrs = gpu(data["ctrs"])
        cl_cands_target = [to_float(cl_cands[i][1]) for i in range(len(cl_cands))]
        ego_fut_traj = [gpu(data['gt_preds'][i][0]) for i in range(len(data['gt_preds']))]
        target_cur_pos = [(torch.matmul(gpu(data["ctrs"][i][1]), gpu(data['rot'][i])) + gpu(data['orig'][i]).view(1, 1, 1, -1))[0, 0, 0, :] for i in range(len(data['gt_preds']))]
        maneuver_out, target_idx = self.maneu_pred(data)
        maneuver_target = [to_float(maneuver_out[i]) for i in target_idx]
        gated_actor_idcs = [torch.tensor(i).unsqueeze(dim=0).cuda() for i in range(len(actor_idcs))]
        gated_actor_ctrs = [actor_ctrs[i][1:2, :] for i in range(len(actor_ctrs))]

        target_fut_traj = self.generator(hidden_target, hidden_ego, cl_cands_target, ego_fut_traj, target_cur_pos, maneuver_target, gated_actor_idcs, gated_actor_ctrs)
        # transform prediction to world coordinates
        for i in range(len(target_fut_traj["reg"])):
            target_fut_traj["reg"][i] = torch.matmul(target_fut_traj["reg"][i], rot[i]) + orig[i].view(
                1, 1, 1, -1
            )

        target_gt_traj = [gpu(data['gt_preds'][i][1, :, :]) for i in range(len(data['gt_preds']))]
        tot_trajectory_fake = [torch.cat([torch.repeat_interleave(target_hist_traj[i].unsqueeze(dim=0), 6, dim=0),
                                          torch.transpose(target_fut_traj["reg"][i].squeeze(), 1, 2)],
                                         dim=2) for i in range(len(target_hist_traj))]
        tot_trajectory_real = [torch.cat([target_hist_traj[i].unsqueeze(dim=0),
                                          torch.transpose(target_gt_traj[i].unsqueeze(dim=0), 1, 2)],
                                         dim=2) for i in range(len(target_hist_traj))]

        dis_real = self.discriminator(tot_trajectory_real, get_hidden=False)
        dis_fake = self.discriminator(tot_trajectory_fake, get_hidden=False)
        hidden_real = self.discriminator(tot_trajectory_real,get_hidden=True)
        hidden_fake = self.discriminator(tot_trajectory_fake,get_hidden=True)

        return target_gt_traj, target_fut_traj, dis_real, dis_fake, hidden_real, hidden_fake, mu_hidden_ego, log_var_hidden_ego


def feat_to_global(targets, rot, orig, ctrs):
    batch_num = len(targets)
    targets_mod = [torch.zeros_like(targets[i])[0, :2, :] for i in range(batch_num)]
    for i in range(batch_num):
        target_cur_pos = ctrs[i]
        targets_mod[i][:, -1] = target_cur_pos
        target_disp = targets[i][0, :2, :]
        for j in range(18, -1, -1):
            targets_mod[i][:, j] = targets_mod[i][:, j + 1] - target_disp[:, j + 1]
        targets_mod[i] = (torch.matmul(torch.inverse(rot[i]), targets_mod[i]).T + orig[i].reshape(-1, 2)).T

    return targets_mod


# target shape = (1, 3, 20)
class EncodeNet(nn.Module):
    def __init__(self, config, args):
        super(EncodeNet, self).__init__()
        self.config = config
        self.actor_net = ActorNet(config)
        self.map_net = MapNet(config)

        self.a2m = A2M(config)
        self.m2m = M2M(config)
        self.m2a = M2A(config)
        self.a2a = A2A(config)

    def forward(self, data):
        actors, actor_idcs, _ = actor_gather(gpu(data["feats"]))

        actors = self.actor_net(actors)
        graph = graph_gather(to_long(gpu(data["graph"])))
        nodes, node_idcs, node_ctrs = self.map_net(graph)
        actor_ctrs = gpu(data["ctrs"])

        nodes = self.a2m(nodes, graph, actors, actor_idcs, actor_ctrs)
        nodes = self.m2m(nodes, graph)
        actors = self.m2a(actors, actor_idcs, actor_ctrs, nodes, node_idcs, node_ctrs)
        actors = self.a2a(actors, actor_idcs, actor_ctrs)
        return actors


class EgoEncodeNet(nn.Module):
    def __init__(self, config):
        super(EgoEncodeNet, self).__init__()
        self.config = config
        self.mu_gen = nn.Linear(config['n_actor'], config['gan_noise_dim'])
        self.log_varience_gen = nn.Linear(config['n_actor'], config['gan_noise_dim'])

    def forward(self, hidden_ego):
        mus = self.mu_gen(hidden_ego)
        log_vars = self.log_varience_gen(hidden_ego)

        return [mus, log_vars]


class GenerateNet(nn.Module):
    def __init__(self, config):
        super(GenerateNet, self).__init__()
        self.config = config

        self.react_net = ReactNet(config)
        self.gating_net = GateNet(config)
        self.pred_net = PredNet(config)

    def forward(self, hidden_target, hidden_ego, cl_cands_target, ego_fut_traj, target_cur_pos, maneuver_target, gated_actor_idcs, gated_actor_ctrs):
        pred_inter = self.react_net(hidden_target, hidden_ego)
        gating_fact = self.gating_net(cl_cands_target, ego_fut_traj, target_cur_pos, maneuver_target)
        gated_actors = torch.sum(torch.mul(gating_fact, torch.cat([pred_inter.unsqueeze(dim=1), hidden_target.unsqueeze(dim=1)], dim=1)), dim=1)
        out = self.pred_net(gated_actors, gated_actor_idcs, gated_actor_ctrs)

        return out


class DiscriminateNet(nn.Module):
    def __init__(self, config):
        super(DiscriminateNet, self).__init__()
        self.config = config
        self.relu6 = nn.ReLU6()
        self.sigmoid = nn.Sigmoid()

        self.discriminator = nn.LSTM(input_size=2,
                                     hidden_size=16,
                                     num_layers=2,
                                     bidirectional=True)
        in_channel = [32, 32, 64]
        out_channel = [32, 64, 64]
        # length = [50, 12, 3, 1]
        kernel_size = [4, 4, 3]
        stride = [4, 4, 3]
        padding = 0
        dilation = 1
        conv1d = []
        for i in range(len(in_channel)):
            net = nn.Conv1d(in_channels=in_channel[i],
                            out_channels=out_channel[i],
                            kernel_size=kernel_size[i],
                            stride=stride[i],
                            padding=padding,
                            dilation=dilation)
            conv1d.append(net)
            conv1d.append(nn.ReLU6())
        self.conv1d = nn.Sequential(*conv1d)
        self.out = nn.Linear(in_channel[-1], 1)


    def forward(self, tot_trajectory, get_hidden):
        # len(tot_trajectory) = batch_num
        # tot_trajectory.shape = (l, 2, 50) l=1 if traj = real, l = 6 if traj = fake
        cat_trajectory = torch.cat(tot_trajectory, dim=0)
        tot_displacement = torch.zeros_like(cat_trajectory)
        tot_displacement[:, :, 1:] = cat_trajectory[:, :, 1:] - cat_trajectory[:, :, :-1]
        tot_displacement = torch.transpose(torch.transpose(tot_displacement, 1, 2), 0, 1)

        seq_emb, _ = self.discriminator(tot_displacement)
        seq_emb = self.relu6(torch.transpose(torch.transpose(seq_emb, 0, 1), 1, 2))

        hid = self.conv1d(seq_emb).squeeze()
        outs = self.sigmoid(self.out(hid))
        if get_hidden:
            return hid
        else:
            return outs


def actor_gather(actors):
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
        count = count + num_actors[i]
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
            out = out + self.lateral[i](outputs[i])

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
        feat = feat + self.seg(graph["feats"])
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
            feat = feat + res
            feat = self.relu(feat)
            res = feat
        return feat, graph["idcs"], graph["ctrs"]


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
            feat = feat + res
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


class ReactNet(nn.Module):
    def __init__(self, config):
        super(ReactNet, self).__init__()
        self.config = config
        self.relu = nn.ReLU()

        n_out = [256, 256, 256, 256]
        blocks = [nn.Linear, nn.Linear, nn.Linear]

        groups = []
        for i in range(len(blocks)):
            group = blocks[i](n_out[i], n_out[i + 1])
            groups.append(group)

        self.inter_pred = nn.ModuleList(groups)

        in_channel = [4, 8, 16, 32, 64]
        out_channel = [8, 16, 32, 64, 128]
        kernel_size = [4, 4, 4, 2, 2]
        stride = [4, 4, 4, 2, 2]
        padding = 0
        dilation = 1
        conv1d = []
        for i in range(len(in_channel)):
            net = nn.Conv1d(in_channels=in_channel[i],
                            out_channels=out_channel[i],
                            kernel_size=kernel_size[i],
                            stride=stride[i],
                            padding=padding,
                            dilation=dilation)
            conv1d.append(net)
            conv1d.append(nn.ReLU6())
        self.conv1d = nn.Sequential(*conv1d)
        self.conv1d_out = nn.Linear(128, 128)

    def forward(self, actors_target, actors_ego):
        cat = torch.cat([actors_target, actors_ego], dim=1)
        cat_tot = [cat.unsqueeze(dim=1)]
        for i in range(len(self.inter_pred)):
            cat = self.inter_pred[i](cat)
            cat = self.relu(cat)
            cat_tot.append(cat.unsqueeze(dim=1))
        cat_tot = torch.cat(cat_tot, dim=1)

        pred_inter = self.conv1d_out(self.conv1d(cat_tot).squeeze())

        return pred_inter


class GateNet(nn.Module):
    def __init__(self, config):
        super(GateNet, self).__init__()
        self.config = config
        self.relu = nn.ReLU()

        in_channel = [3, 9, 27, 81]
        out_channel = [9, 27, 81, 128]
        kernel_size = [2, 2, 2, 2, 2]
        stride = [2, 2, 2, 2, 2]
        padding = 0
        dilation = 1

        conv1d = []
        for i in range(len(in_channel)):
            net = nn.Conv1d(in_channels=in_channel[i],
                            out_channels=out_channel[i],
                            kernel_size=kernel_size[i],
                            stride=stride[i],
                            padding=padding,
                            dilation=dilation)
            conv1d.append(net)
            if i < len(in_channel):
                conv1d.append(nn.ReLU6())
            else:
                conv1d.append(nn.Sigmoid())
        self.gate = nn.Sequential(*conv1d)

    def forward(self, cl_cands_target, ego_fut_traj, target_cur_pos, maneuver_target):
        batch_num = len(cl_cands_target)
        gating = []
        for i in range(batch_num):
            target_cl = cl_cands_target[i]
            target_cl = self.cl_filter(target_cl)
            ego_fut = ego_fut_traj[i]
            dist_mat = [torch.cdist(ego_fut, target_cl[j]) for j in range(len(target_cl))]
            dist_to_target_cl = [torch.min(dist_mat[j]) for j in range(len(target_cl))]
            if min(dist_to_target_cl) > config["inter_dist_thres"]:
                gating_tmp = torch.cat([torch.zeros(1, 128), torch.ones(1, 128)], dim=0).unsqueeze(dim=0)
                gating_tmp = gating_tmp.cuda()
                gating.append(gating_tmp)
            else:
                dist_to = torch.cat([torch.min(dist_mat[j], dim=1)[0].unsqueeze(dim=0).unsqueeze(dim=0) for j in range(len(target_cl))], dim=0)
                delta_x = torch.repeat_interleave((ego_fut[:, 0] - target_cur_pos[i][0]).unsqueeze(dim=0).unsqueeze(dim=0), len(target_cl), dim=0)
                delta_y = torch.repeat_interleave((ego_fut[:, 1] - target_cur_pos[i][1]).unsqueeze(dim=0).unsqueeze(dim=0), len(target_cl), dim=0)
                gating_in = torch.cat([delta_x, delta_y, dist_to], dim=1)
                gating_out = self.gate(gating_in).squeeze(dim=-1)
                gating_tmp = torch.cat([gating_out[j:j + 1, :] * maneuver_target[i][j] for j in range(gating_out.shape[0])], dim=0)
                gating_tmp = torch.sum(gating_tmp, dim=0).unsqueeze(dim=0)
                gating_tmp = torch.cat([gating_tmp, torch.ones_like(gating_tmp) - gating_tmp]).unsqueeze(dim=0)
                gating.append(gating_tmp)
        gating = torch.cat(gating, dim=0)
        return gating

    def cl_filter(self, target_cl):
        val_idx = []
        for i in range(len(target_cl)):
            if i == 0:
                val_idx.append(i)
            else:
                val_check = []
                for j in val_idx:
                    val_len = min(target_cl[i].shape[0], target_cl[j].shape[0], 50)
                    val_check.append((target_cl[i][:val_len, :] == target_cl[j][:val_len, :]).all().unsqueeze(dim=0))
                val_check = torch.cat(val_check, dim=0)
                if val_check.any():
                    pass
                else:
                    val_idx.append(i)
        return [target_cl[i] for i in val_idx]


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
            agts = agts + res
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
            hi_count = hi_count + len(agt_idcs[i])
            wi_count = wi_count + len(ctx_idcs[i])
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
        agts = agts + res
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
        self.reg_loss = nn.SmoothL1Loss(reduction="sum")
        self.cls_loss = nn.BCELoss(reduction='sum')
        self.hidden_loss = nn.L1Loss(reduction='sum')
        self.pred_loss = PredLoss(config)

    def forward(self, target_gt_traj, target_fut_traj, dis_real, dis_fake, hidden_real, hidden_fake, mu_hidden_ego, log_var_hidden_ego, data):
        target_gt_traj_cat = torch.cat([torch.repeat_interleave(target_gt_traj[i].unsqueeze(dim=0), 6, dim=0) for i in range(len(target_gt_traj))], dim=0)
        target_fut_traj_cat = torch.cat([target_fut_traj['reg'][i].squeeze() for i in range(len(target_gt_traj))])
        l1loss_trajectory = self.reg_loss(target_gt_traj_cat, target_fut_traj_cat)
        hidden_real_cat = torch.cat([torch.repeat_interleave(hidden_real[i:i+1,:], 6, dim=0) for i in range(len(target_gt_traj))], dim=0)
        MAELoss_layer = self.hidden_loss(hidden_real_cat, hidden_fake)

        kl_loss = torch.sum(-0.5 * torch.sum(-log_var_hidden_ego.exp() - torch.pow(mu_hidden_ego,2) + log_var_hidden_ego + 1, 1))

        bce_dis_real = self.cls_loss(dis_real, torch.ones_like(dis_real))
        bce_dis_fake = self.cls_loss(dis_fake, torch.zeros_like(dis_fake))

        bce_gen_real = self.cls_loss(dis_real, torch.zeros_like(dis_real))
        bce_gen_fake = self.cls_loss(dis_fake, torch.ones_like(dis_fake))

        gt_preds = [gpu(data["gt_preds"])[i][1:2, :, :] for i in range(len(data["gt_preds"]))]
        has_preds = [gpu(data["has_preds"])[i][1:2, :] for i in range(len(data["has_preds"]))]
        loss_out = self.pred_loss(target_fut_traj, gt_preds, has_preds)
        loss_out["loss"] = loss_out["cls_loss"] / (
                loss_out["num_cls"] + 1e-10
        ) + loss_out["reg_loss"] / (loss_out["num_reg"] + 1e-10)

        loss_out['l1loss_trajectory'] = l1loss_trajectory
        loss_out['MAELoss_layer'] = MAELoss_layer
        loss_out['kl_loss'] = kl_loss
        loss_out['bce_gen_fake'] = bce_gen_fake
        loss_out['bce_gen_real'] = bce_gen_real
        loss_out['bce_dis_fake'] = bce_dis_fake
        loss_out['bce_dis_real'] = bce_dis_real

        return loss_out


class PostProcess(nn.Module):
    def __init__(self, config, args):
        super(PostProcess, self).__init__()
        self.config = config
        self.args = args

    def forward(self, out, data):
        post_out = dict()
        post_out["preds"] = [x.detach().cpu().numpy() for x in out["reg"]]
        post_out["gt_preds"] = [x[1:2].numpy() for x in data["gt_preds"]]
        post_out["has_preds"] = [x[1:2].numpy() for x in data["has_preds"]]
        return post_out

    def append(self, metrics: Dict, loss_out: Dict, post_out: Optional[Dict[str, List[ndarray]]] = None) -> Dict:
        if len(metrics.keys()) == 0:
            for key in loss_out:
                if key != "loss":
                    metrics[key] = 0.0

            for key in post_out:
                metrics[key] = []

        for key in loss_out:
            if key == "loss":
                continue
            if isinstance(loss_out[key], torch.Tensor):
                metrics[key] += loss_out[key].item()
            else:
                metrics[key] += loss_out[key]

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

        cls = metrics["cls_loss"] / (metrics["num_cls"] + 1e-10)
        reg = metrics["reg_loss"] / (metrics["num_reg"] + 1e-10)
        l1loss_trajectory = metrics["l1loss_trajectory"] / (len(metrics["preds"]))
        MAELoss_layer = metrics["MAELoss_layer"] / (len(metrics["preds"]))
        kl_loss = metrics["kl_loss"] / (len(metrics["preds"]))
        bce_gen_fake = metrics["bce_gen_fake"] / (len(metrics["preds"]))
        bce_gen_real = metrics["bce_gen_real"] / (len(metrics["preds"]))
        bce_dis_fake = metrics["bce_dis_fake"] / (len(metrics["preds"]))
        bce_dis_real = metrics["bce_dis_real"] / (len(metrics["preds"]))

        loss_encoder = kl_loss + l1loss_trajectory
        loss_discriminator = bce_dis_fake + bce_dis_real
        loss_generator = 0.2 * l1loss_trajectory + (1.0 - 0.2) * (bce_gen_fake + bce_gen_real)

        preds = np.concatenate(metrics["preds"], 0)
        gt_preds = np.concatenate(metrics["gt_preds"], 0)
        has_preds = np.concatenate(metrics["has_preds"], 0)
        ade1, fde1, ade, fde, min_idcs = pred_metrics(preds, gt_preds, has_preds)

        print(
            "loss_encoder %2.4f, loss_discriminator %2.4f, loss_generator %2.4f, ade1 %2.4f, fde1 %2.4f, ade %2.4f, fde %2.4f"
            % (loss_encoder, loss_discriminator, loss_generator, ade1, fde1, ade, fde)
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
    net = lanegcn_vanilla_gan(config, args)
    params_ego_enc = [(name, param) for name, param in net.ego_encoder.named_parameters()]
    params_gen = [(name, param) for name, param in net.generator.named_parameters()]
    params_dis = [(name, param) for name, param in net.discriminator.named_parameters()]

    parama0 = [p for n, p in params_ego_enc]
    params1 = [p for n, p in params_gen]
    params2 = [p for n, p in params_dis]
    opt = [Optimizer(parama0, config), Optimizer(params1, config), Optimizer(params2, config)]
    loss = Loss(config).cuda()
    params = [params_ego_enc, params_gen, params_dis]

    net = net.cuda()
    # post_process = PostProcess(config,args).cuda()
    post_process = PostProcess(config,args).cuda()
    config["save_dir"] = os.path.join(
        config["save_dir"], args.case
    )
    return config, ArgoDataset, collate_fn, net, loss, post_process, opt, params
