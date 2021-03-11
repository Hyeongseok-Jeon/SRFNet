import numpy as np
import os
import sys
from fractions import gcd
from numbers import Number

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from LaneGCN.layers import Conv1d, Res1d, Linear, LinearRes, Null
from LaneGCN.utils import gpu, to_long, Optimizer, StepLR

from data import ArgoDataset, collate_fn

from numpy import float64, ndarray
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from SRFNet.layer import GraphAttentionLayer


class Net(nn.Module):
    """
    Lane Graph Network contains following components:
        1. ActorNet: a 1D CNN to process the trajectory input
        2. MapNet: LaneGraphCNN to learn structured map representations
           from vectorized map data
        3. Actor-Map Fusion Cycle: fuse the information between actor nodes
           and lane nodes:
            a. A2M: introduces real-time traffic information to
                lane nodes, such as blockage or usage of the lanes
            b. M2M:  updates lane node features by propagating the
                traffic information over lane graphs
            c. M2A: fuses updated map features with real-time traffic
                information back to actors
            d. A2A: handles the interaction between actors and produces
                the output actor features
        4. PredNet: prediction header for motion forecasting using
           feature from A2A
    """

    def __init__(self, config):
        super(Net, self).__init__()
        self.config = config

        self.actor_net = ActorNet(config)
        self.map_net = MapNet(config)
        self.tempAtt_net = TempAttNet(config)
        self.SRF_net = SRF_net(config)
        self.pred_net = PredNet(config)

    def forward(self, data: Dict) -> Dict[str, List[Tensor]]:
        # construct actor feature
        actors = torch.cat(gpu(data['actors']), dim=0)
        actor_ctrs = gpu(data["ctrs"])
        actor_idcs = []
        veh_calc = 0
        for i in range(len(data['actor_idcs'])):
            actor_idcs.append(gpu(data['actor_idcs'][i][0] + veh_calc))
            veh_calc += len(data['actor_idcs'][i][0])
        actors = self.actor_net(actors, actor_idcs)

        # construct map features
        graph = to_long(gpu(map_graph_gather(data)))
        nodes, node_idcs, node_ctrs = self.map_net(graph)

        # concat actor and map features
        actor_graph = actor_graph_gather(actors, nodes, actor_idcs, self.config, graph, data)

        # get temporal attention matrix
        [one_step_feats, adjs] = self.tempAtt_net(actor_graph)
        veh_calc = 0
        reaction_to_ego = []
        for i in range(len(data['actor_idcs'])):
            reaction_to_ego.append(adjs[:, veh_calc:veh_calc + len(data['actor_idcs'][i][0]), veh_calc, :])
            veh_calc += len(data['actor_idcs'][i][0])

        reaction_to_ego = torch.transpose(torch.transpose(torch.cat(reaction_to_ego, dim=1), 0, 1), 1, 2)
        reaction_hidden = self.SRF_net(reaction_to_ego)

        # get ego future traj
        ego_fut = [torch.repeat_interleave(gpu(data['ego_feats'][i]), len(data['actor_idcs'][i][0]), dim=0) for i in range(len(data['actor_idcs']))]
        ego_fut = torch.cat(ego_fut, dim=0)
        ego_feat = [self.actor_net(torch.transpose(ego_fut[:, ts - 20:ts, :], 1, 2), actor_idcs) for ts in range(20, 50)]

        # prediction
        actors_cat = torch.cat([actors[i][:, -1, :] for i in range(len(actors))], dim=0) + reaction_hidden
        out = self.pred_net(actors_cat, actor_idcs, actor_ctrs)
        if self.config["reactive"]:

        rot, orig = gpu(data["rot"]), gpu(data["orig"])
        # transform prediction to world coordinates
        for i in range(len(out["reg"])):
            out["reg"][i] = torch.matmul(out["reg"][i], rot[i]) + orig[i].view(
                1, 1, 1, -1
            )
        return out


def map_graph_gather(data):
    map_node_cnt = [0]
    veh_cnt = 0
    for i in range(len(data['actor_idcs']) - 1):
        map_node_cnt.append(len(data['graph_mod'][i]['idcs'][0]) + veh_cnt)
        veh_cnt += len(data['graph_mod'][i]['idcs'][0])

    graph_mod = dict()
    idcs = [data['graph_mod'][i]['idcs'][0] + map_node_cnt[i] for i in range(len(data['actor_idcs']))]
    ctrs = [data['graph_mod'][i]['ctrs'][0] for i in range(len(data['actor_idcs']))]
    feats = torch.cat([data['graph_mod'][i]['feats'] for i in range(len(data['actor_idcs']))], dim=0)
    turn = torch.cat([data['graph_mod'][i]['turn'] for i in range(len(data['actor_idcs']))], dim=0)
    control = torch.cat([data['graph_mod'][i]['control'] for i in range(len(data['actor_idcs']))], dim=0)
    intersect = torch.cat([data['graph_mod'][i]['intersect'] for i in range(len(data['actor_idcs']))], dim=0)

    graph_mod['idcs'] = idcs
    graph_mod['ctrs'] = ctrs
    graph_mod['feats'] = feats
    graph_mod['turn'] = turn
    graph_mod['control'] = control
    graph_mod['intersect'] = intersect
    graph_mod['pre'] = []
    graph_mod['suc'] = []
    for j in range(6):
        for i in range(len(data['actor_idcs'])):
            if i == 0:
                pre_tmp_u = data['graph_mod'][i]['pre'][j]['u'] + map_node_cnt[i]
                pre_tmp_v = data['graph_mod'][i]['pre'][j]['v'] + map_node_cnt[i]
                suc_tmp_u = data['graph_mod'][i]['suc'][j]['u'] + map_node_cnt[i]
                suc_tmp_v = data['graph_mod'][i]['suc'][j]['v'] + map_node_cnt[i]
                if j < 2:
                    left_tmp_u = data['graph_mod'][i]['left']['u'] + map_node_cnt[i]
                    left_tmp_v = data['graph_mod'][i]['left']['v'] + map_node_cnt[i]
                    right_tmp_u = data['graph_mod'][i]['right']['u'] + map_node_cnt[i]
                    right_tmp_v = data['graph_mod'][i]['right']['v'] + map_node_cnt[i]
            else:
                pre_tmp_u = torch.cat((pre_tmp_u, data['graph_mod'][i]['pre'][j]['u'] + map_node_cnt[i]))
                pre_tmp_v = torch.cat((pre_tmp_v, data['graph_mod'][i]['pre'][j]['v'] + map_node_cnt[i]))
                suc_tmp_u = torch.cat((suc_tmp_u, data['graph_mod'][i]['suc'][j]['u'] + map_node_cnt[i]))
                suc_tmp_v = torch.cat((suc_tmp_v, data['graph_mod'][i]['suc'][j]['v'] + map_node_cnt[i]))
                if j < 2:
                    left_tmp_u = torch.cat((left_tmp_u, data['graph_mod'][i]['left']['u'] + map_node_cnt[i]))
                    left_tmp_v = torch.cat((left_tmp_v, data['graph_mod'][i]['left']['v'] + map_node_cnt[i]))
                    right_tmp_u = torch.cat((right_tmp_u, data['graph_mod'][i]['right']['u'] + map_node_cnt[i]))
                    right_tmp_v = torch.cat((right_tmp_v, data['graph_mod'][i]['right']['v'] + map_node_cnt[i]))

        graph_mod['pre'].append({'u': pre_tmp_u, 'v': pre_tmp_v})
        graph_mod['suc'].append({'u': suc_tmp_u, 'v': suc_tmp_v})
    graph_mod['left'] = {'u': left_tmp_u, 'v': left_tmp_v}
    graph_mod['right'] = {'u': right_tmp_u, 'v': right_tmp_v}
    return graph_mod


def actor_graph_gather(actors, nodes, actor_idcs, config, graph, data):
    batch_size = len(actors)
    tot_veh_num = actor_idcs[-1][-1] + 1
    node_feat_mask = gpu(torch.zeros(size=(tot_veh_num, 20, config['n_actor'] + config['n_map'])))
    gen_num = 0
    adj_mask = gpu(torch.zeros(size=(tot_veh_num, tot_veh_num, config['n_actor'])))

    maps = []
    for i in range(batch_size):
        idx = data['feats'][i][:, :, 2:].cuda()
        idx = torch.repeat_interleave(idx, config['n_actor'], dim=-1)
        map_node_idx = graph["idcs"][i][data['nearest_ctrs_hist'][i].long()]
        map_node = nodes[map_node_idx]
        maps.append(map_node * idx)

    for i in range(batch_size):
        node_feat_mask[gen_num:gen_num + actors[i].shape[0], :, :config['n_actor']] = actors[i]
        node_feat_mask[gen_num:gen_num + actors[i].shape[0], :, config['n_actor']:] = maps[i]
        adj_mask[gen_num:gen_num + actors[i].shape[0], gen_num:gen_num + actors[i].shape[0], :] = 1
        gen_num += actors[i].shape[0]

    actor_graph = dict()
    actor_graph["node_feat"] = node_feat_mask
    actor_graph["adj_mask"] = adj_mask
    return actor_graph


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

    def forward(self, actors: Tensor, actor_idcs: List) -> Tensor:
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
        if actors.shape[0] != actor_idcs[-1][-1]+1:
            out = self.reform_out(out, actor_idcs)

        return out

    def reform_out(self, out, actor_idcs):
        veh_cnt = 0
        actors_batch = []
        for i in range(len(actor_idcs)):
            actors_mini_batch = []
            for j in range(20):
                actors_mini_batch.append(out[veh_cnt:veh_cnt + len(actor_idcs[i]), :].unsqueeze(dim=1))
                veh_cnt = veh_cnt + len(actor_idcs[i])
            actors_batch.append(torch.cat(actors_mini_batch, dim=1))

        return actors_batch


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


class TempAttNet(nn.Module):
    def __init__(self, config):
        super(TempAttNet, self).__init__()
        self.config = config
        self.W = nn.Parameter(torch.empty(size=(config["n_actor"], config["n_actor"])))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.GAT = GAT(config["n_actor"] + config["n_map"],
                       config["n_actor"],
                       config["GAT_dropout"],
                       config["GAT_Leakyrelu_alpha"],
                       nTime=20,
                       training=config["training"])

    def forward(self, actor_graph):
        out = self.GAT(actor_graph, self.W)
        feats = torch.cat(out[0], dim=0)
        adjs = torch.cat(out[1], dim=0)

        return [feats, adjs]


class SRF_net(nn.Module):
    def __init__(self, config):
        super(SRF_net, self).__init__()
        self.config = config

        conv = []
        for i in range(config["SRF_conv_num"]):
            conv.append(
                nn.Sequential(
                    nn.Conv1d(in_channels=config['n_actor'],
                              out_channels=config['n_actor'],
                              kernel_size=5,
                              stride=1,
                              padding=2),
                    nn.ReLU(inplace=True)
                )
            )
        self.conv1d = nn.Sequential(*conv)

        out = []
        for i in range(4):
            out.append(
                nn.Sequential(
                    nn.Conv1d(in_channels=config['n_actor'],
                              out_channels=config['n_actor'],
                              kernel_size=4,
                              stride=2,
                              padding=1),
                    nn.ReLU(inplace=True)
                )
            )
        self.out = nn.Sequential(*out)

    def forward(self, feats):
        x = self.conv1d(feats)
        out = self.out(x).squeeze()

        return out


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
    def __init__(self, nfeat, nhid, dropout, alpha, nTime, training):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        self.training = training
        self.nTime = nTime
        self.input_layer = nn.Linear(nfeat, nhid)
        self.attentions = [GraphAttentionLayer(nhid, nhid, dropout=dropout, alpha=alpha, concat=True, training=self.training) for _ in range(nTime)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

    def forward(self, actor_graph, share_weight):
        x = actor_graph['node_feat'][:, 0, :]
        adj = actor_graph['adj_mask']
        x = F.elu(self.input_layer(x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = [att([x, adj], share_weight) for att in self.attentions]
        feats = [F.dropout(x[i][0].unsqueeze(dim=0), self.dropout, training=self.training) for i in range(self.nTime)]
        adjs = [x[i][1].unsqueeze(dim=0) for i in range(self.nTime)]

        return [feats, adjs]


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

# TODO: need to consider global position of the vehicles in GAT
