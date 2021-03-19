import numpy as np
import os
import sys
from fractions import gcd
from numbers import Number

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from LaneGCN.layers import Conv1d, Res1d, Linear, LinearRes, Null
from LaneGCN.utils import gpu, to_long, Optimizer, StepLR, cpu
from numpy import ndarray

from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from SRFNet.layer import GraphAttentionLayer, GraphAttentionLayer_time_serial, GAT_SRF
import time


class model_case_0(nn.Module):
    def __init__(self, config):
        super(model_case_0, self).__init__()
        self.config = config
        self.actor_net = ActorNet(config)
        self.pred_net = PredNet(config)

    def forward(self, data):
        data =data.copy()
        actor_ctrs = gpu(data['actor_ctrs'], gpu_id=self.config['gpu_id'])
        actor_idcs_init = gpu(data['actor_idcs'], gpu_id=self.config['gpu_id'])
        actors_hidden = gpu(data['actors_hidden'], gpu_id=self.config['gpu_id'])
        nodes = gpu(data['nodes'], gpu_id=self.config['gpu_id'])
        graph_idcs = gpu(data['graph_idcs'], gpu_id=self.config['gpu_id'])
        ego_feat = gpu(data['ego_feat'], gpu_id=self.config['gpu_id'])
        feats = gpu(data['feats'], gpu_id=self.config['gpu_id'])
        nearest_ctrs_hist = gpu(data['nearest_ctrs_hist'], gpu_id=self.config['gpu_id'])
        rot = gpu(data['rot'], gpu_id=self.config['gpu_id'])
        orig = gpu(data['orig'], gpu_id=self.config['gpu_id'])
        gt_preds = gpu(data['gt_preds'], gpu_id=self.config['gpu_id'])
        has_preds = gpu(data['has_preds'], gpu_id=self.config['gpu_id'])
        ego_feat_calc = gpu(data['ego_feat_calc'], gpu_id=self.config['gpu_id'])
        actors = gpu(data['actors'], gpu_id=self.config['gpu_id'])
        graph_mod = gpu(data['graph_mod'], gpu_id=self.config['gpu_id'])

        actor_idcs = []
        veh_calc = 0
        for i in range(len(actor_idcs_init)):
            actor_idcs.append(actor_idcs_init[i] + veh_calc)
            veh_calc += len(actor_idcs_init[i])

        actors = torch.cat(actors, dim=0)
        actor_ctrs = actor_ctrs

        actors = self.actor_net(actors, actor_idcs)
        actors_cat = torch.cat([actors[i][:, -1, :] for i in range(len(actors))], dim=0)
        out_non_interact = self.pred_net(actors_cat, actor_idcs, actor_ctrs)
        out_non_interact = self.get_world_cord(out_non_interact, rot, orig)

        return out_non_interact, gt_preds, has_preds

    def get_world_cord(self, out, rot, orig):
        for i in range(len(out["reg"])):
            out["reg"][i] = torch.matmul(out["reg"][i], rot[i][0]) + orig[i][0].view(
                1, 1, 1, -1
            )
        return out


class model_case_1(nn.Module):
    def __init__(self, config):
        super(model_case_1, self).__init__()
        self.config = config
        self.actor_net = ActorNet(config)
        self.map_net = MapNet(config)
        self.fusion_net = FusionNet(config)
        self.pred_net = PredNet(config)
        self.inter_pred_net = ReactPredNet(config)

    def forward(self, inputs):
        actor_ctrs = inputs[0]
        actor_idcs_init = inputs[1]
        graph_idcs = inputs[4]
        feats = inputs[6]
        nearest_ctrs_hist = inputs[7]
        actors = inputs[11]
        data = inputs[12]
        graph_mod = inputs[13]
        rot = inputs[8]
        orig = inputs[9]

        actor_idcs = []
        veh_calc = 0
        for i in range(len(actor_idcs_init)):
            actor_idcs.append(actor_idcs_init[i] + veh_calc)
            veh_calc += len(actor_idcs_init[i])

        actors = torch.cat(actors, dim=0)

        actors = self.actor_net(actors, actor_idcs)
        actors_inter_cat = torch.cat(actors, dim=0)
        actors_pred_cat = torch.cat([actors[i][:, -1, :] for i in range(len(actors))], dim=0)

        graph = to_long(gpu(self.map_graph_gather(data), self.config['gpu_id']))
        nodes, node_idcs, node_ctrs = self.map_net(graph)

        for i in range(len(nearest_ctrs_hist)):
            if i == 0:
                nearest_ctrs_hist[i] = nearest_ctrs_hist[i]
            else:
                nearest_ctrs_hist[i] = nearest_ctrs_hist[i] + np.sum(np.asarray([len(graph_idcs[j]) for j in range(i)]))
        nearest_ctrs_cat = torch.cat(nearest_ctrs_hist, dim=0)
        graph_adjs = self.graph_adj(actors_inter_cat, nearest_ctrs_cat, nodes)

        interaction_mod = self.fusion_net(actors_inter_cat, graph_adjs)

        out_non_interact = self.pred_net(actors_pred_cat, actor_idcs, actor_ctrs)
        out_non_interact = self.get_world_cord(out_non_interact, rot, orig)

        out_sur_interact = self.inter_pred_net(interaction_mod, actor_idcs, actor_ctrs)
        out_sur_interact = self.get_world_cord(out_sur_interact, rot, [[torch.zeros_like(orig[i][0])] for i in range(len(orig))])
        for i in range(len(out_sur_interact['reg'])):
            out_sur_interact['reg'][i] = 3 * out_sur_interact['reg'][i]

        out = dict()
        out['cls'] = out_non_interact['cls']
        out['reg'] = [out_non_interact['reg'][i] + torch.repeat_interleave(out_sur_interact['reg'][i], 6, dim=1) for i in range(len(out_non_interact['reg']))]
        return [out_non_interact, out_sur_interact, out]

    def get_world_cord(self, out, rot, orig):
        for i in range(len(out["reg"])):
            out["reg"][i] = torch.matmul(out["reg"][i], rot[i][0]) + orig[i][0].view(
                1, 1, 1, -1
            )
        return out

    def graph_adj(self, actors_cat, nearest_ctrs_cat, nodes):
        edge = torch.cat([nodes[nearest_ctrs_cat[i].long()].unsqueeze(dim=0) for i in range(actors_cat.shape[0])], dim=0)

        return edge

    def map_graph_gather(self, data):
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
        if actors.shape[0] != actor_idcs[-1][-1] + 1:
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
                [temp.new().long().resize_(0) for x in graph["idcs"]],
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
            if i == 0:
                out_tmp, [c0, h0] = self.GAT_lstm(actors_inter_cat[:, 5 * (i + 1) - 1, :], graph_adjs[:, 0, :], [c0, h0])
            else:
                out_tmp, [c0, h0] = self.GAT_lstm(actors_inter_cat[:, 5 * (i + 1) - 1, :], graph_adjs[:, 5 * i - 1, :], [c0, h0])

            out.append(F.sigmoid(out_tmp))
        out = torch.cat([out[i].unsqueeze(dim=2) for i in range(len(out))], dim=2)
        out = self.conv1d(out).squeeze()
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


class ReactPredNet(nn.Module):
    """
    Final motion forecasting with Linear Residual block
    """

    def __init__(self, config):
        super(ReactPredNet, self).__init__()
        self.config = config
        norm = "GN"
        ng = 1

        n_actor = config["n_actor"]

        pred = []
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

        dest_ctrs = reg[:, :, -1].detach()
        feats = self.att_dest(actors, torch.cat(actor_ctrs, 0), dest_ctrs)
        cls = self.cls(feats).view(-1, 1)

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
        loss_out = self.pred_loss(out, gpu(data["gt_preds"], self.config['gpu_id']), gpu(data["has_preds"], self.config['gpu_id']))
        loss_out["loss"] = loss_out["cls_loss"] / (
                loss_out["num_cls"] + 1e-10
        ) + loss_out["reg_loss"] / (loss_out["num_reg"] + 1e-10)
        return loss_out


class Loss_light(nn.Module):
    def __init__(self, config):
        super(Loss_light, self).__init__()
        self.config = config
        self.pred_loss = PredLoss(config)

    def forward(self, out, gt_preds, has_preds):
        loss_out = self.pred_loss(out, gt_preds, has_preds)
        loss_out["loss"] = loss_out["cls_loss"] / (
                loss_out["num_cls"] + 1e-10
        ) + loss_out["reg_loss"] / (loss_out["num_reg"] + 1e-10)
        return loss_out


class PostProcess(nn.Module):
    def __init__(self, config):
        super(PostProcess, self).__init__()
        self.config = config

    def forward(self, out, gt_preds, has_preds):
        post_out = dict()
        post_out["preds"] = [x[1:2].detach().cpu().numpy() for x in out["reg"]]
        post_out["gt_preds"] = [x[1:2].numpy() for x in cpu(gt_preds)]
        post_out["has_preds"] = [x[1:2].numpy() for x in cpu(has_preds)]
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
        loss = cls + reg

        preds = np.concatenate(metrics["preds"], 0)
        gt_preds = np.concatenate(metrics["gt_preds"], 0)
        has_preds = np.concatenate(metrics["has_preds"], 0)
        ade1, fde1, ade, fde, min_idcs = pred_metrics(preds, gt_preds, has_preds)

        print(
            "loss %2.4f %2.4f %2.4f, ade1 %2.4f, fde1 %2.4f, ade %2.4f, fde %2.4f"
            % (loss, cls, reg, ade1, fde1, ade, fde)
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


class inter_loss(nn.Module):
    def __init__(self, config):
        super(inter_loss, self).__init__()
        self.config = config
        self.pred_loss = torch.nn.L1Loss()

    def forward(self, out, gt, has_preds):
        for i in range(len(out['reg'])):
            out['reg'][i].squeeze()[~torch.repeat_interleave(has_preds[i].unsqueeze(dim=-1), 2, dim=-1)] = 0
            for j in range(6):
                gt[i][:,j,:,:][~torch.repeat_interleave(has_preds[i].unsqueeze(dim=-1), 2, dim=-1)] = 0

        loss = self.pred_loss(torch.cat(out['reg'], dim=0), torch.cat(gt, dim=0))

        return loss

# TODO: need to consider global position of the vehicles in GAT
