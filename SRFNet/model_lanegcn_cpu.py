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

from SRFNet.utils import gpu, to_long, Optimizer, StepLR, to_float

from SRFNet.layers import Conv1d, Res1d, Linear, LinearRes, Null, GraphAttentionLayer, GraphAttentionLayer_time_serial, GAT_SRF
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union


### end of config ###

class lanegcn(nn.Module):
    def __init__(self, config):
        super(lanegcn, self).__init__()
        self.config = config

        self.actor_net = ActorNet(config)
        self.map_net = MapNet(config)

        self.a2m = A2M(config)
        self.m2m = M2M(config)
        self.m2a = M2A(config)
        self.a2a = A2A(config)
        self.pred_net = PredNet(config)

    def forward(self, data: Dict) -> Dict[str, List[Tensor]]:
        # construct actor feature

        actors, actor_idcs, _ = actor_gather([torch.from_numpy(data['feats'][i]) for i in range(len(data['feats']))])
        actor_ctrs = [torch.from_numpy(data['ctrs'][i]) for i in range(len(data['ctrs']))]

        '''
        actors : N x 3 x 20 (N : number of vehicles in every batches)
        '''

        actors = self.actor_net(actors)

        # construct map features
        graph = graph_gather(to_long(data["graph"]))
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
        rot, orig = [torch.from_numpy(data['rot'][i]) for i in range(len(data['rot']))], [torch.from_numpy(data['orig'][i]) for i in range(len(data['orig']))]
        # transform prediction to world coordinates
        for i in range(len(out["reg"])):
            out["reg"][i] = torch.matmul(out["reg"][i], rot[i]) + orig[i].view(
                1, 1, 1, -1
            )
        return [out]


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
        idcs = torch.arange(count, count + graphs[i]["num_nodes"])
        node_idcs.append(idcs)
        count = count + graphs[i]["num_nodes"]

    graph = dict()
    graph["idcs"] = node_idcs
    graph["ctrs"] = [torch.from_numpy(x["ctrs"]) for x in graphs]

    for key in ["feats", "turn", "control", "intersect"]:
        graph[key] = torch.cat([torch.from_numpy(x[key]) for x in graphs], 0)

    for k1 in ["pre", "suc"]:
        graph[k1] = []
        for i in range(len(graphs[0]["pre"])):
            graph[k1].append(dict())
            for k2 in ["u", "v"]:
                graph[k1][i][k2] = torch.cat([torch.from_numpy(graphs[j][k1][i][k2] + counts[j]) for j in range(batch_size)], 0)

    for k1 in ["left", "right"]:
        graph[k1] = dict()
        for k2 in ["u", "v"]:
            temp = [graphs[i][k1][k2] + counts[i] for i in range(batch_size)]
            temp = [torch.from_numpy(x) if x.ndim > 0 else graph["pre"][0]["u"].new().resize_(0) for x in temp]
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
            if min(dist_to_target_cl) > self.config["inter_dist_thres"]:
                gating_tmp = torch.cat([torch.zeros(1, 128), torch.ones(1, 128)], dim=0).unsqueeze(dim=0)
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


'''         
late fusion 방식일 경우에는 delta x, y를 prediction 결과에 대해서 진행
mid fusion 방식일 경우에는 delta x, y에 대해서 cur_pos에 대해서 진행
'''

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
                        graph[k1][k2]["u"].type(torch.int64),
                        self.fuse[key][i](feat[to_long(graph[k1][k2]["v"])]),
                    )

            if len(graph["left"]["u"] > 0):
                temp.index_add_(
                    0,
                    graph["left"]["u"].type(torch.int64),
                    self.fuse["left"][i](feat[to_long(graph["left"]["v"])]),
                )
            if len(graph["right"]["u"] > 0):
                temp.index_add_(
                    0,
                    graph["right"]["u"].type(torch.int64),
                    self.fuse["right"][i](feat[to_long(graph["right"]["v"])]),
                )

            feat = self.fuse["norm"][i](temp)
            feat = self.relu(feat)

            feat = self.fuse["ctr2"][i](feat)
            feat += res
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
                        graph[k1][k2]["u"].type(torch.int64),
                        self.fuse[key][i](feat[to_long(graph[k1][k2]["v"])]),
                    )

            if len(graph["left"]["u"] > 0):
                temp.index_add_(
                    0,
                    graph["left"]["u"].type(torch.int64),
                    self.fuse["left"][i](feat[to_long(graph["left"]["v"])]),
                )
            if len(graph["right"]["u"] > 0):
                temp.index_add_(
                    0,
                    graph["right"]["u"].type(torch.int64),
                    self.fuse["right"][i](feat[to_long(graph["right"]["v"])]),
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


def get_model(config):
    net = lanegcn(config)

    return net
