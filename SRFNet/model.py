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
from SRFNet.layer import GraphAttentionLayer, GraphAttentionLayer_time_serial
import time


class Net_SRF(nn.Module):
    def __init__(self, config):
        super(Net_SRF, self).__init__()
        self.config = config

        self.tempAtt_net = TempAttNet(config)


    def forward(self, inputs):
        # construct actor feature
        actor_ctrs = inputs[0]
        actor_idcs_init = inputs[1]
        actor_idcs = []
        veh_calc = 0
        for i in range(len(actor_idcs_init)):
            actor_idcs.append(actor_idcs_init[i] + veh_calc)
            veh_calc += len(actor_idcs_init[i])
        actors = inputs[2]
        nodes = inputs[3]
        graph_idcs = inputs[4]
        ego_feat = inputs[5]
        feats = inputs[6]
        nearest_ctrs_hist = inputs[7]
        rot = inputs[8]
        orig = inputs[9]
        ego_feat_calc = inputs[10]
        # concat actor and map features
        actor_graph = self.actor_graph_gather(actors, nodes, actor_idcs, self.config, graph_idcs, [feats, nearest_ctrs_hist])
        # get temporal attention matrix

        [one_step_feats, adjs] = self.tempAtt_net(actor_graph)

        return one_step_feats

    def actor_graph_gather(self, actors, nodes, actor_idcs, config, graph_idcs, data):
        batch_size = len(actors)
        tot_veh_num = actor_idcs[-1][-1] + 1
        node_feat_mask = gpu(torch.zeros(size=(tot_veh_num, 20, config['n_actor'] + config['n_map'])), config['gpu_id'])
        gen_num = 0
        adj_mask = gpu(torch.zeros(size=(tot_veh_num, tot_veh_num, config['n_actor'])), config['gpu_id'])

        maps = []
        for i in range(batch_size):
            idx = data[0][i][:, :, 2:]
            idx = torch.repeat_interleave(idx, config['n_actor'], dim=-1)
            map_node_idx = graph_idcs[i][data[1][i].long()]
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


class Net_min(nn.Module):
    def __init__(self, config):
        super(Net_min, self).__init__()
        self.config = config

        self.tempAtt_net = TempAttNet(config)
        self.SRF_net = SRFNet(config)
        self.pred_net = PredNet(config)
        self.reaction_net = ReactNet(config)

    def forward(self, inputs):
        # construct actor feature
        actor_ctrs = inputs[0]
        actor_idcs_init = inputs[1]
        actor_idcs = []
        veh_calc = 0
        for i in range(len(actor_idcs_init)):
            actor_idcs.append(actor_idcs_init[i] + veh_calc)
            veh_calc += len(actor_idcs_init[i])
        actors = inputs[2]
        nodes = inputs[3]
        graph_idcs = inputs[4]
        ego_feat = inputs[5]
        feats = inputs[6]
        nearest_ctrs_hist = inputs[7]
        rot = inputs[8]
        orig = inputs[9]
        ego_feat_calc = inputs[10]
        # concat actor and map features
        actor_graph = self.actor_graph_gather(actors, nodes, actor_idcs, self.config, graph_idcs, [feats, nearest_ctrs_hist])
        # get temporal attention matrix

        [one_step_feats, adjs] = self.tempAtt_net(actor_graph)

        veh_calc = 0
        reaction_to_veh = []
        for i in range(len(actor_idcs)):
            reaction_to_i = []
            for j in range(len(actor_idcs[i])):
                reaction_to_i.append(adjs[:, veh_calc + j, veh_calc:veh_calc + len(actor_idcs[i]), :])
            reaction_to_veh.append(reaction_to_i)
            veh_calc += len(actor_idcs[i])

        # reaction to i_th vehicle from j_th vehicle in k_th batch : reaction_to_veh[k][i][:,j,:]
        reaction_hiddens = self.SRF_net(reaction_to_veh)

        reaction_hidden = []
        for i in range(len(reaction_hiddens)):
            reaction_hidden = reaction_hidden + reaction_hiddens[i]

        # prediction
        actors_cat = torch.cat([actors[i][:, -1, :] for i in range(len(actors))], dim=0)

        if self.config['interaction'] == 'none':
            out_non_interact = self.pred_net(actors_cat, actor_idcs, actor_ctrs)
            out_non_interact = self.get_world_cord(out_non_interact, rot, orig)
            return out_non_interact
        else:
            actors_cat_sur_inter = torch.zeros_like(actors_cat)
            for i in range(len(actor_idcs)):
                actor_base_hid = actors_cat[actor_idcs[i]]
                for j in range(len(actor_idcs[i])):
                    inter_feat = reaction_hidden[actor_idcs[i][j]]
                    actors_cat_sur_inter[actor_idcs[i]] = actor_base_hid * inter_feat
            out_sur_interact = self.pred_net(actors_cat_sur_inter, actor_idcs, actor_ctrs)
            if self.config['interaction'] == 'sur':
                out_sur_interact = self.get_world_cord(out_sur_interact, rot, orig)
                return out_sur_interact
            elif self.config['interaction'] == 'ego':
                out_ego_interact_tmp = self.reaction_net(reaction_hidden, ego_feat_calc, actor_idcs)
                out_ego_interact = dict()
                out_ego_interact['cls'] = out_sur_interact['cls']
                out_ego_interact['reg'] = [out_sur_interact['reg'][i] + out_ego_interact_tmp[i] for i in range(len(actor_idcs))]
                out_ego_interact = self.get_world_cord(out_ego_interact, rot, orig)
                return out_ego_interact

    def get_world_cord(self, out, rot, orig):
        for i in range(len(out["reg"])):
            out["reg"][i] = torch.matmul(out["reg"][i], rot[i][0]) + orig[i][0].view(
                1, 1, 1, -1
            )
        return out

    def actor_graph_gather(self, actors, nodes, actor_idcs, config, graph_idcs, data):
        batch_size = len(actors)
        tot_veh_num = actor_idcs[-1][-1] + 1
        node_feat_mask = gpu(torch.zeros(size=(tot_veh_num, 20, config['n_actor'] + config['n_map'])), config['gpu_id'])
        gen_num = 0
        adj_mask = gpu(torch.zeros(size=(tot_veh_num, tot_veh_num, config['n_actor'])), config['gpu_id'])

        maps = []
        for i in range(batch_size):
            idx = data[0][i][:, :, 2:]
            idx = torch.repeat_interleave(idx, config['n_actor'], dim=-1)
            map_node_idx = graph_idcs[i][data[1][i].long()]
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


class Net_full(nn.Module):
    def __init__(self, config):
        super(Net_full, self).__init__()
        self.config = config

        self.actor_net = ActorNet(config)
        self.map_net = MapNet(config)
        self.tempAtt_net = TempAttNet_original(config)
        self.SRF_net = SRFNet(config)
        self.pred_net = PredNet(config)
        self.reaction_net = ReactNet(config)

    def forward(self, data: Dict) -> Dict[str, List[Tensor]]:
        # construct actor feature
        init_time = time.time()
        actors = torch.cat(gpu(data['actors'], self.config['gpu_id']), dim=0)
        actor_ctrs = gpu(data["ctrs"], self.config['gpu_id'])
        actor_idcs = []
        veh_calc = 0
        for i in range(len(data['actor_idcs'])):
            actor_idcs.append(gpu(data['actor_idcs'][i][0] + veh_calc, self.config['gpu_id']))
            veh_calc += len(data['actor_idcs'][i][0])
        actors = self.actor_net(actors, actor_idcs)
        # print('a')
        # print(time.time()- init_time)
        # construct map features
        graph = to_long(gpu(map_graph_gather(data), self.config['gpu_id']))
        nodes, node_idcs, node_ctrs = self.map_net(graph)
        # print('b')
        # print(time.time()- init_time)

        # concat actor and map features
        actor_graph = actor_graph_gather(actors, nodes, actor_idcs, self.config, graph["idcs"], data)

        # get temporal attention matrix
        [one_step_feats, adjs] = self.tempAtt_net(actor_graph)
        veh_calc = 0
        reaction_to_veh = []
        for i in range(len(data['actor_idcs'])):
            reaction_to_i = []
            for j in range(len(data['actor_idcs'][i][0])):
                reaction_to_i.append(adjs[:, veh_calc + j, veh_calc:veh_calc + len(data['actor_idcs'][i][0]), :])
            reaction_to_veh.append(reaction_to_i)
            veh_calc += len(data['actor_idcs'][i][0])
        # print('c')
        # print(time.time() - init_time)

        ## reaction to i_th vehicle from j_th vehicle in k_th batch : reaction_to_veh[k][i][:,j,:]
        reaction_hiddens = self.SRF_net(reaction_to_veh)
        reaction_hidden = []
        for i in range(len(reaction_hiddens)):
            reaction_hidden = reaction_hidden + reaction_hiddens[i]
        # print('d')
        # print(time.time()- init_time)

        # get ego future traj and calc interaction
        ego_fut = [torch.repeat_interleave(gpu(data['ego_feats'][i], self.config['gpu_id']), len(data['actor_idcs'][i][0]), dim=0) for i in range(len(data['actor_idcs']))]
        ego_fut = torch.cat(ego_fut, dim=0)
        ego_feat = [self.actor_net(torch.transpose(ego_fut[:, ts - 20:ts, :], 1, 2), actor_idcs) for ts in range(20, 50)]

        # prediction
        actors_cat = torch.cat([actors[i][:, -1, :] for i in range(len(actors))], dim=0)
        actors_cat_sur_inter = torch.zeros_like(actors_cat)
        for i in range(len(actor_idcs)):
            actor_base_hid = actors_cat[actor_idcs[i]]
            for j in range(len(actor_idcs[i])):
                inter_feat = reaction_hidden[actor_idcs[i][j]]
                actors_cat_sur_inter[actor_idcs[i]] = actor_base_hid * inter_feat

        out_non_interact = self.pred_net(actors_cat, actor_idcs, actor_ctrs)
        out_sur_interact = self.pred_net(actors_cat_sur_inter, actor_idcs, actor_ctrs)
        out_ego_interact_tmp = self.reaction_net(reaction_hidden, ego_feat, actor_idcs)
        out_ego_interact = dict()
        out_ego_interact['cls'] = out_sur_interact['cls']
        out_ego_interact['reg'] = [out_sur_interact['reg'][i] + out_ego_interact_tmp[i] for i in range(len(actor_idcs))]
        # print('e')
        # print(time.time()- init_time)

        out_non_interact = self.get_world_cord(out_non_interact, data)
        out_sur_interact = self.get_world_cord(out_sur_interact, data)
        out_ego_interact = self.get_world_cord(out_ego_interact, data)
        return [one_step_feats, out_non_interact, out_sur_interact, out_ego_interact]

    def get_world_cord(self, out, data):
        rot, orig = gpu(data["rot"], self.config['gpu_id']), gpu(data["orig"], self.config['gpu_id'])
        for i in range(len(out["reg"])):
            out["reg"][i] = torch.matmul(out["reg"][i], rot[i]) + orig[i].view(
                1, 1, 1, -1
            )
        return out


class pre_net(nn.Module):
    def __init__(self, config):
        super(pre_net, self).__init__()
        self.config = config

        self.actor_net = ActorNet(config)
        self.map_net = MapNet(config)

    def forward(self, data: Dict) -> Dict[str, List[Tensor]]:
        # construct actor feature
        actors = torch.cat(gpu(data['actors'], self.config['gpu_id']), dim=0)
        actor_ctrs = gpu(data["ctrs"], self.config['gpu_id'])
        actor_idcs = []
        veh_calc = 0
        for i in range(len(data['actor_idcs'])):
            actor_idcs.append(gpu(data['actor_idcs'][i][0] + veh_calc, self.config['gpu_id']))
            veh_calc += len(data['actor_idcs'][i][0])
        actors_hidden = self.actor_net(actors, actor_idcs)

        graph = to_long(gpu(map_graph_gather(data), self.config['gpu_id']))
        nodes, node_idcs, node_ctrs = self.map_net(graph)

        ego_fut = [torch.repeat_interleave(gpu(data['ego_feats'][i], self.config['gpu_id']), len(data['actor_idcs'][i][0]), dim=0) for i in range(len(data['actor_idcs']))]
        ego_fut = torch.cat(ego_fut, dim=0)
        ego_feat = [self.actor_net(torch.transpose(ego_fut[:, ts - 20:ts, :], 1, 2), actor_idcs) for ts in range(20, 50)]

        return [actors_hidden, nodes, node_idcs, node_ctrs, graph["idcs"], ego_feat]


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


def actor_graph_gather(actors, nodes, actor_idcs, config, graph_idcs, data):
    batch_size = len(actors)
    tot_veh_num = actor_idcs[-1][-1] + 1
    node_feat_mask = gpu(torch.zeros(size=(tot_veh_num, 20, config['n_actor'] + config['n_map'])), config['gpu_id'])
    gen_num = 0
    adj_mask = gpu(torch.zeros(size=(tot_veh_num, tot_veh_num, config['n_actor'])), config['gpu_id'])

    maps = []
    for i in range(batch_size):
        idx = data['feats'][i][:, :, 2:].cuda(config['gpu_id'])
        idx = torch.repeat_interleave(idx, config['n_actor'], dim=-1)
        map_node_idx = graph_idcs[i][data['nearest_ctrs_hist'][i].long()]
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
                       training=config["training"],
                       config=self.config)

    def forward(self, actor_graph):
        out = self.GAT(actor_graph, self.W)
        feats = torch.cat(out[0], dim=0)
        adjs = torch.cat(out[1], dim=0)

        return [feats, adjs]


class TempAttNet_original(nn.Module):
    def __init__(self, config):
        super(TempAttNet_original, self).__init__()
        self.config = config
        self.W = nn.Parameter(torch.empty(size=(config["n_actor"], config["n_actor"])))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.GAT = GAT_original(config["n_actor"] + config["n_map"],
                                config["n_actor"],
                                config["GAT_dropout"],
                                config["GAT_Leakyrelu_alpha"],
                                nTime=20,
                                training=config["training"],
                                config=self.config)

    def forward(self, actor_graph):
        out = self.GAT(actor_graph, self.W)
        feats = torch.cat(out[0], dim=0)
        adjs = torch.cat(out[1], dim=0)

        return [feats, adjs]


class SRFNet(nn.Module):
    def __init__(self, config):
        super(SRFNet, self).__init__()
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
        hidden = []
        for k in range(len(feats)):
            veh_num = len(feats[k])
            in_data = torch.cat([torch.transpose(torch.transpose(feats[k][i], 0, 1), 1, 2) for i in range(len(feats[k]))], dim=0)
            out_tmp = self.out(self.conv1d(in_data)).squeeze()
            to_i = [out_tmp[veh_num*i:veh_num*(i+1),:] for i in range(veh_num)]
            hidden.append(to_i)
        return hidden


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


class ReactNet(nn.Module):
    def __init__(self, config):
        super(ReactNet, self).__init__()
        self.config = config
        norm = "GN"
        ng = 1

        n_actor = config["n_actor"]

        self.react_pred = nn.LSTM(input_size=config["n_actor"],
                                  hidden_size=config["n_actor"],
                                  num_layers=1,
                                  bias=True,
                                  batch_first=False,
                                  dropout=0,
                                  bidirectional=False)

        pred = []
        for i in range(config["num_mods"]):
            pred.append(
                nn.Sequential(
                    LinearRes(n_actor, n_actor, norm=norm, ng=ng),
                    nn.Linear(n_actor, 2),
                )
            )
        self.pred = nn.ModuleList(pred)

    def forward(self, reaction_hidden, ego_feat, actor_idcs):
        reaction_against_ego = torch.cat([x[0, :].unsqueeze(dim=0) for x in reaction_hidden], dim=0).unsqueeze(dim=0)
        ego_feat_time_cat = torch.cat([x.unsqueeze(dim=0) for x in ego_feat])

        react_feat, (hn, cn) = self.react_pred(ego_feat_time_cat, (torch.zeros_like(reaction_against_ego), reaction_against_ego))

        feat_tmp = react_feat.view(30 * react_feat.shape[1], 128)
        reacts = []
        for i in range(len(self.pred)):
            reacts_cand = self.pred[i](feat_tmp).view(30, react_feat.shape[1], 2)
            reacts.append(reacts_cand.unsqueeze(dim=0))
        reacts = torch.cat(reacts, dim=0)

        reacts_mod = []
        for i in range(len(actor_idcs)):
            reacts_mod.append(torch.transpose(torch.transpose(reacts[:, :, actor_idcs[i], :], 1, 2), 0, 1))

        return reacts_mod


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
    def __init__(self, nfeat, nhid, dropout, alpha, nTime, training, config):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.config = config
        self.dropout = dropout
        self.training = training
        self.nTime = nTime
        self.input_layer = nn.Linear(nfeat, nhid)
        self.attentions = GraphAttentionLayer_time_serial(config, nhid, nhid, dropout=dropout, alpha=alpha, concat=True, training=self.training, nTime=nTime)

        # self.attentions = [GraphAttentionLayer(nhid, nhid, dropout=dropout, alpha=alpha, concat=True, training=self.training) for _ in range(nTime)]
        # for i, attention in enumerate(self.attentions):
        #     self.add_module('attention_{}'.format(i), attention)

    def forward(self, actor_graph, share_weight):
        # x = actor_graph['node_feat'][:, 0, :]
        # adj = actor_graph['adj_mask']
        # x = F.elu(input_layer(x))
        # x = F.dropout(x, dropout, training=training)
        # x = [att([x, adj], share_weight) for att in attentions]
        # feats = [F.dropout(x[i][0].unsqueeze(dim=0), dropout, training=training) for i in range(nTime)]
        # adjs = [x[i][1].unsqueeze(dim=0) for i in range(nTime)]

        x = torch.cat([actor_graph['node_feat'][:, i, :] for i in range(self.nTime)], dim=0)
        adj = actor_graph['adj_mask']
        x = F.elu(self.input_layer(x))
        x = F.dropout(x, self.dropout, training=self.training)
        feats, adjs = self.attentions([x, adj], share_weight)
        adjs = [adjs[i].unsqueeze(dim=0) for i in range(len(adjs))]
        return [feats, adjs]


class GAT_original(nn.Module):
    def __init__(self, nfeat, nhid, dropout, alpha, nTime, training, config):
        """Dense version of GAT."""
        super(GAT_original, self).__init__()
        self.config = config
        self.dropout = dropout
        self.training = training
        self.nTime = nTime
        self.input_layer = nn.Linear(nfeat, nhid)
        # self.attentions = GraphAttentionLayer_time_serial(config, nhid, nhid, dropout=dropout, alpha=alpha, concat=True, training=self.training, nTime=nTime)

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

        # x = torch.cat([actor_graph['node_feat'][:, i, :] for i in range(self.nTime)], dim=0)
        # adj = actor_graph['adj_mask']
        # x = F.elu(self.input_layer(x))
        # x = F.dropout(x, self.dropout, training=self.training)
        # feats, adjs = self.attentions([x, adj], share_weight)
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

# TODO: need to consider global position of the vehicles in GAT
