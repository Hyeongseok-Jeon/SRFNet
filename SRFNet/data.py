# Copyright (c) 2020 Uber Technologies, Inc.
# Please check LICENSE for more detail


import numpy as np
import torch
from torch.utils.data import Dataset
from scipy import sparse
import os
import copy
from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
from argoverse.map_representation.map_api import ArgoverseMap
from skimage.transform import rotate
from SRFNet.model_lanegcn_cpu import get_model


class ArgoDataset(Dataset):
    def __init__(self, split, config, train=True):
        self.config = config
        self.train = train
        root_path = os.getcwd()
        base_model = get_model(config)
        pre_trained_weight = torch.load(os.path.join(root_path, "../LaneGCN/pre_trained") + '/36.000.ckpt')
        pretrained_dict = pre_trained_weight['state_dict']
        new_model_dict = base_model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in new_model_dict}
        new_model_dict.update(pretrained_dict)
        base_model.load_state_dict(new_model_dict)
        self.base_model = base_model.cpu()
        if 'preprocess' in config and config['preprocess']:
            if train:
                self.split = np.load(self.config['preprocess_train'], allow_pickle=True)
            else:
                self.split = np.load(self.config['preprocess_val'], allow_pickle=True)
        else:
            self.avl = ArgoverseForecastingLoader(split)
            self.am = ArgoverseMap()

        if 'raster' in config and config['raster']:
            # TODO: DELETE
            self.map_query = MapQuery(config['map_scale'])

    def reform(self, ego_fut_traj, cl_cands_target, init_pred_global):
        reformed = []
        for i in range(len(cl_cands_target)):
            if len(cl_cands_target[i]) == 1:
                cl_cand = torch.from_numpy(cl_cands_target[i][0])
            else:
                cl_cand = self.get_nearest_cl([torch.from_numpy(cl_cands_target[i][j]) for j in range(len(cl_cands_target[i]))], init_pred_global['reg'][i])
            cl_cand = cl_cand[:100]
            ego_fut = ego_fut_traj[i][0]
            mask = torch.zeros_like(ego_fut)
            mask = torch.repeat_interleave(torch.repeat_interleave(mask, 4, dim=0)[:100, 0].unsqueeze(dim=0), 30, dim=0)
            ego_pos = ego_fut
            ego_dists = torch.norm(torch.repeat_interleave(cl_cand.unsqueeze(dim=1), 30, dim=1) - ego_pos, dim=2)
            ego_dist, ego_idx = torch.min(ego_dists, dim=0)

            sur_pos = init_pred_global['reg'][i][0, :, :, :]
            sur_dists = [torch.norm(torch.repeat_interleave(cl_cand.unsqueeze(dim=1), 30, dim=1) - sur_pos[i, :], dim=2) for i in range(6)]
            sur_dist, sur_idx = [], []
            for j in range(6):
                sur_dist_tmp, sur_idx_tmp = torch.min(sur_dists[j], dim=0)
                sur_dist.append(sur_dist_tmp)
                sur_idx.append(sur_idx_tmp)
            sur_min_idx = min([min(sur_idx[ss]) for ss in range(6)])
            min_idx = min(sur_min_idx, min(ego_idx))

            sur_disp = [cl_cand[sur_idx[i]] - sur_pos[i] for i in range(6)]
            sur_feat_1 = [mask.clone() for _ in range(6)]
            for jj in range(6):
                sur_feat_1[jj][np.arange(30), sur_idx[jj] - min_idx] = 1
            sur_feat = torch.cat([torch.cat([sur_feat_1[i], sur_disp[i]], dim=1).unsqueeze(dim=0) for i in range(6)], dim=0)

            ego_disp = cl_cand[ego_idx] - ego_pos
            ego_feat_1 = mask.clone()
            ego_feat_1[np.arange(30), ego_idx - min_idx] = 1
            ego_feat = torch.repeat_interleave(torch.cat([ego_feat_1, ego_disp], dim=1).unsqueeze(dim=0), 6, dim=0)
            feat = torch.cat([ego_feat, sur_feat], dim=-1)
            reformed.append(feat)
        return reformed

    def get_nearest_cl(self, cl_cands_target_tmp, init_pred_global_tmp):
        dist = []
        for i in range(len(cl_cands_target_tmp)):
            dist_tmp = []
            cl_tmp = cl_cands_target_tmp[i]
            cl_tmp_dense = self.get_cl_dense(cl_tmp)
            for j in range(30):
                tmp = cl_tmp_dense - init_pred_global_tmp[0, :, j:j + 1, :]
                tmps = torch.sqrt(tmp[:, :, 0] ** 2 + tmp[:, :, 1] ** 2)
                dist_tmp_tmp = torch.mean(torch.min(tmps, dim=1)[0]).unsqueeze(dim=0)
                dist_tmp.append(dist_tmp_tmp)
            dist_tmp = torch.cat(dist_tmp)
            dist.append(torch.mean(dist_tmp).unsqueeze(dim=0))
        dist = torch.cat(dist)
        return cl_cands_target_tmp[torch.argmin(dist)]

    def get_cl_dense(self, cl_tmp):
        cl_mod = torch.zeros_like(torch.repeat_interleave(cl_tmp, 4, dim=0))
        for i in range(cl_tmp.shape[0]):
            if i == cl_tmp.shape[0] - 1:
                cl_mod[4 * i, :] = cl_tmp[i, :]
            else:
                cl_mod[4 * i, :] = cl_tmp[i, :]
                cl_mod[4 * i + 1, :] = cl_tmp[i, :] + 1 * (cl_tmp[i + 1, :] - cl_tmp[i, :]) / 4
                cl_mod[4 * i + 2, :] = cl_tmp[i, :] + 2 * (cl_tmp[i + 1, :] - cl_tmp[i, :]) / 4
                cl_mod[4 * i + 3, :] = cl_tmp[i, :] + 3 * (cl_tmp[i + 1, :] - cl_tmp[i, :]) / 4
        cl_mod = cl_mod[:-3, :]
        return torch.repeat_interleave(cl_mod.unsqueeze(dim=0), 6, dim=0)

    def __getitem__(self, idx):
        if 'preprocess' in self.config and self.config['preprocess']:
            data = self.split[idx]

            if self.train and self.config['rot_aug']:
                new_data = dict()
                for key in ['city', 'orig', 'gt_preds', 'has_preds', 'file_name', 'cl_cands', 'cl_cands_mod', 'gt_cl_cands']:
                    if key in data:
                        new_data[key] = ref_copy(data[key])

                dt = np.random.rand() * self.config['rot_size']  # np.pi * 2.0
                theta = data['theta'] + dt
                new_data['theta'] = theta
                new_data['rot'] = np.asarray([
                    [np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]], np.float32)

                rot = np.asarray([
                    [np.cos(-dt), -np.sin(-dt)],
                    [np.sin(-dt), np.cos(-dt)]], np.float32)
                new_data['feats'] = data['feats'].copy()
                new_data['feats'][:, :, :2] = np.matmul(new_data['feats'][:, :, :2], rot)
                new_data['ego_feats'] = data['ego_feats'].copy()
                new_data['ego_feats'][:, :, :2] = np.matmul(new_data['ego_feats'][:, :, :2], rot)
                new_data['ctrs'] = np.matmul(data['ctrs'], rot)

                graph = dict()
                for key in ['num_nodes', 'turn', 'control', 'intersect', 'pre', 'suc', 'lane_idcs', 'left_pairs', 'right_pairs', 'left', 'right']:
                    graph[key] = ref_copy(data['graph'][key])
                graph['ctrs'] = np.matmul(data['graph']['ctrs'], rot)
                graph['feats'] = np.matmul(data['graph']['feats'], rot)
                graph['ego_feats'] = np.matmul(data['ego_feats']['feats'], rot)
                new_data['graph'] = graph
                data = get_ctrs_idx(new_data)
            else:
                new_data = dict()
                for key in ['city', 'orig', 'gt_preds', 'has_preds', 'theta', 'rot', 'feats', 'ego_feats', 'ctrs', 'graph', 'file_name', 'cl_cands', 'cl_cands_mod', 'gt_cl_cands']:
                    if key in data:
                        new_data[key] = ref_copy(data[key])
                data = get_ctrs_idx(new_data)

            if 'raster' in self.config and self.config['raster']:
                data.pop('graph')
                x_min, x_max, y_min, y_max = self.config['pred_range']
                cx, cy = data['orig']

                region = [cx + x_min, cx + x_max, cy + y_min, cy + y_max]
                raster = self.map_query.query(region, data['theta'], data['city'])

                data['raster'] = raster

            for key in data.keys():
                data[key] = [data[key]]

            with torch.no_grad():
                init_pred_global = self.base_model(data)
            init_pred_global_con = init_pred_global[0]
            init_pred_global_con['reg'] = [init_pred_global_con['reg'][i][1:2, :, :, :] for i in range(len(init_pred_global_con['reg']))]

            ego_fut_traj = [torch.from_numpy(data['gt_preds'][i][0:1, :, :]) for i in range(len(data['gt_preds']))]

            cl_cands = (data['cl_cands'])
            cl_cands_target = [cl_cands[i][1] for i in range(len(cl_cands))]
            hid = self.reform(ego_fut_traj, cl_cands_target, init_pred_global_con)
            for key in data.keys():
                data[key] = data[key][0]
            data['data'] = hid
            data['init_pred_global'] = init_pred_global
            data['init_pred_global_con'] = init_pred_global_con

            return data

        data = self.read_argo_data(idx)
        data = self.get_obj_feats(data)
        data['idx'] = idx

        if 'raster' in self.config and self.config['raster']:
            x_min, x_max, y_min, y_max = self.config['pred_range']
            cx, cy = data['orig']

            region = [cx + x_min, cx + x_max, cy + y_min, cy + y_max]
            raster = self.map_query.query(region, data['theta'], data['city'])

            data['raster'] = raster
            return data

        data['graph'] = self.get_lane_graph(data)
        cl_cands_mod, gt_cl_cands = self.cl_cands_gather(data['cl_cands'], data['feats'], data)
        data['cl_cands_mod'] = cl_cands_mod
        data['gt_cl_cands'] = gt_cl_cands
        return data

    def __len__(self):
        if 'preprocess' in self.config and self.config['preprocess']:
            return len(self.split)
        else:
            return len(self.avl)

    def cl_cands_gather(self, cl_cands, actors_in_batch, data):
        cl_mask = np.zeros(shape=(50, 4))
        cl_tot = []

        veh_in_batch = len(cl_cands)
        veh_feats = self.disp_to_global(actors_in_batch, data)
        feats = data['feats']
        future_traj = data['gt_preds']
        cl_in_batch = []
        gt_in_batch = []
        for j in range(veh_in_batch):
            num_path_cands_of_veh = len(cl_cands[j])
            cl_veh = []
            cl_global = []
            traj = veh_feats[j]
            valid_num = np.sum(feats[j][:, 2], dtype=int)
            fut_traj = future_traj[j, :, :]
            traj_valid = traj[:valid_num, :]
            if num_path_cands_of_veh > 0:
                for k in range(num_path_cands_of_veh):
                    cl_mask_tmp = cl_mask.copy()
                    cl_rots = np.matmul(data['rot'], (cl_cands[j][k] - data['orig'].reshape(-1, 2)).T).T + data['orig'].reshape(-1, 2)
                    cl_mods = cl_rots[1:] - cl_rots[:-1]
                    cl_mask_tmp[1:cl_rots.shape[0], 0] = cl_mods[:min(cl_rots.shape[0] - 1, 49), 0]
                    cl_mask_tmp[1:cl_rots.shape[0], 1] = cl_mods[:min(cl_rots.shape[0] - 1, 49), 1]
                    idxs = [-1, -1]
                    for l in range(min(cl_rots.shape[0], 50)):
                        idxs[0] = idxs[1]
                        idxs[1] = np.argmin(np.linalg.norm(traj_valid - cl_rots[l, :], axis=1))
                        if idxs[0] == idxs[1] and idxs[0] == valid_num - 1:
                            cl_mask_tmp[l, 3] = 0
                            cl_mask_tmp[l, 2] = 0
                        else:
                            cl_mask_tmp[l, 3] = 1
                            cl_mask_tmp[l, 2] = np.min(np.linalg.norm(traj_valid - cl_rots[l, :], axis=1))
                    cl_mask_tmp = np.expand_dims(cl_mask_tmp, axis=0)
                    if k == 0:
                        cl_global.append(cl_cands[j][k])
                        cl_veh.append(cl_mask_tmp)
                    else:
                        val_check = [cl_mask_tmp == cl_veh[i] for i in range(len(cl_veh))]
                        tot_check = [val_check[i].all() for i in range(len(val_check))]
                        if np.asarray(tot_check).any():
                            pass
                        else:
                            cl_global.append(cl_cands[j][k])
                            cl_veh.append(cl_mask_tmp)
                gt_mask = np.zeros(len(cl_global))
                dist_to_cl_fut = []
                for k in range(len(cl_global)):
                    cl_cand = cl_global[k]
                    dist_to_cl_fut.append(np.mean([np.min(np.linalg.norm(cl_cand - fut_traj[i], axis=1)) for i in range(30)]))
                gt_mask[np.argmin(dist_to_cl_fut)] = 1
            else:
                gt_mask = np.zeros(1)
                cl_veh = [np.expand_dims(cl_mask.copy(), axis=0)]
            cl_in_batch.append(np.concatenate(cl_veh, axis=0))
            gt_in_batch.append(gt_mask)

        return cl_in_batch, gt_in_batch

    def disp_to_global(self, actors_in_batch, data):
        mask = np.zeros_like(actors_in_batch)[:, :, :2]
        cur_pos = data['orig'] + data['ctrs']
        mask[:, -1, :] = cur_pos
        for i in range(18, -1, -1):
            mask[:, i, :] = mask[:, i + 1, :] - data['feats'][:, i + 1, :2]
        return mask

    def read_argo_data(self, idx):
        city = copy.deepcopy(self.avl[idx].city)

        """TIMESTAMP,TRACK_ID,OBJECT_TYPE,X,Y,CITY_NAME"""
        df = copy.deepcopy(self.avl[idx].seq_df)

        agt_ts = np.sort(np.unique(df['TIMESTAMP'].values))
        mapping = dict()
        for i, ts in enumerate(agt_ts):
            mapping[ts] = i

        trajs = np.concatenate((
            df.X.to_numpy().reshape(-1, 1),
            df.Y.to_numpy().reshape(-1, 1)), 1)

        steps = [mapping[x] for x in df['TIMESTAMP'].values]
        steps = np.asarray(steps, np.int64)

        objs = df.groupby(['TRACK_ID', 'OBJECT_TYPE']).groups
        keys = list(objs.keys())
        obj_type = [x[1] for x in keys]

        agt_idx = obj_type.index('AGENT')
        idcs = objs[keys[agt_idx]]

        agt_traj = trajs[idcs]
        agt_step = steps[idcs]

        del keys[agt_idx]
        ctx_trajs, ctx_steps = [], []
        for key in keys:
            idcs = objs[key]
            ctx_trajs.append(trajs[idcs])
            ctx_steps.append(steps[idcs])

        data = dict()
        data['city'] = city
        data['trajs'] = [agt_traj] + ctx_trajs
        data['steps'] = [agt_step] + ctx_steps
        return data

    def get_obj_feats(self, data):
        orig = data['trajs'][0][19].copy().astype(np.float32)

        if self.train and self.config['rot_aug']:
            theta = np.random.rand() * np.pi * 2.0
        else:
            pre = data['trajs'][0][18] - orig
            theta = np.pi - np.arctan2(pre[1], pre[0])

        rot = np.asarray([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]], np.float32)

        feats, ctrs, gt_preds, has_preds = [], [], [], []
        for traj, step in zip(data['trajs'], data['steps']):
            if 19 not in step:
                continue

            gt_pred = np.zeros((30, 2), np.float32)
            has_pred = np.zeros(30, np.bool)
            future_mask = np.logical_and(step >= 20, step < 50)
            post_step = step[future_mask] - 20
            post_traj = traj[future_mask]
            gt_pred[post_step] = post_traj
            has_pred[post_step] = 1

            obs_mask = step < 20
            step = step[obs_mask]
            traj = traj[obs_mask]
            idcs = step.argsort()
            step = step[idcs]
            traj = traj[idcs]

            for i in range(len(step)):
                if step[i] == 19 - (len(step) - 1) + i:
                    break
            step = step[i:]
            traj = traj[i:]

            feat = np.zeros((20, 3), np.float32)
            feat[step, :2] = np.matmul(rot, (traj - orig.reshape(-1, 2)).T).T
            feat[step, 2] = 1.0

            x_min, x_max, y_min, y_max = self.config['pred_range']
            if feat[-1, 0] < x_min or feat[-1, 0] > x_max or feat[-1, 1] < y_min or feat[-1, 1] > y_max:
                continue

            ctrs.append(feat[-1, :2].copy())
            feat[1:, :2] -= feat[:-1, :2]
            feat[step[0], :2] = 0
            feats.append(feat)
            gt_preds.append(gt_pred)
            has_preds.append(has_pred)

        feats = np.asarray(feats, np.float32)
        ctrs = np.asarray(ctrs, np.float32)
        gt_preds = np.asarray(gt_preds, np.float32)
        has_preds = np.asarray(has_preds, np.bool)

        data['feats'] = feats
        data['ctrs'] = ctrs
        data['orig'] = orig
        data['theta'] = theta
        data['rot'] = rot
        data['gt_preds'] = gt_preds
        data['has_preds'] = has_preds
        return data

    def get_lane_graph(self, data):
        """Get a rectangle area defined by pred_range."""
        x_min, x_max, y_min, y_max = self.config['pred_range']
        radius = max(abs(x_min), abs(x_max)) + max(abs(y_min), abs(y_max))
        lane_ids = self.am.get_lane_ids_in_xy_bbox(data['orig'][0], data['orig'][1], data['city'], radius)
        lane_ids = copy.deepcopy(lane_ids)

        lanes = dict()
        for lane_id in lane_ids:
            lane = self.am.city_lane_centerlines_dict[data['city']][lane_id]
            lane = copy.deepcopy(lane)
            centerline = np.matmul(data['rot'], (lane.centerline - data['orig'].reshape(-1, 2)).T).T
            x, y = centerline[:, 0], centerline[:, 1]
            if x.max() < x_min or x.min() > x_max or y.max() < y_min or y.min() > y_max:
                continue
            else:
                """Getting polygons requires original centerline"""
                polygon = self.am.get_lane_segment_polygon(lane_id, data['city'])
                polygon = copy.deepcopy(polygon)
                lane.centerline = centerline
                lane.polygon = np.matmul(data['rot'], (polygon[:, :2] - data['orig'].reshape(-1, 2)).T).T
                lanes[lane_id] = lane

        lane_ids = list(lanes.keys())
        ctrs, feats, turn, control, intersect = [], [], [], [], []
        for lane_id in lane_ids:
            lane = lanes[lane_id]
            ctrln = lane.centerline
            num_segs = len(ctrln) - 1

            ctrs.append(np.asarray((ctrln[:-1] + ctrln[1:]) / 2.0, np.float32))
            feats.append(np.asarray(ctrln[1:] - ctrln[:-1], np.float32))

            x = np.zeros((num_segs, 2), np.float32)
            if lane.turn_direction == 'LEFT':
                x[:, 0] = 1
            elif lane.turn_direction == 'RIGHT':
                x[:, 1] = 1
            else:
                pass
            turn.append(x)

            control.append(lane.has_traffic_control * np.ones(num_segs, np.float32))
            intersect.append(lane.is_intersection * np.ones(num_segs, np.float32))

        node_idcs = []
        count = 0
        for i, ctr in enumerate(ctrs):
            node_idcs.append(range(count, count + len(ctr)))
            count += len(ctr)
        num_nodes = count

        pre, suc = dict(), dict()
        for key in ['u', 'v']:
            pre[key], suc[key] = [], []
        for i, lane_id in enumerate(lane_ids):
            lane = lanes[lane_id]
            idcs = node_idcs[i]

            pre['u'] += idcs[1:]
            pre['v'] += idcs[:-1]
            if lane.predecessors is not None:
                for nbr_id in lane.predecessors:
                    if nbr_id in lane_ids:
                        j = lane_ids.index(nbr_id)
                        pre['u'].append(idcs[0])
                        pre['v'].append(node_idcs[j][-1])

            suc['u'] += idcs[:-1]
            suc['v'] += idcs[1:]
            if lane.successors is not None:
                for nbr_id in lane.successors:
                    if nbr_id in lane_ids:
                        j = lane_ids.index(nbr_id)
                        suc['u'].append(idcs[-1])
                        suc['v'].append(node_idcs[j][0])

        lane_idcs = []
        for i, idcs in enumerate(node_idcs):
            lane_idcs.append(i * np.ones(len(idcs), np.int64))
        lane_idcs = np.concatenate(lane_idcs, 0)

        pre_pairs, suc_pairs, left_pairs, right_pairs = [], [], [], []
        for i, lane_id in enumerate(lane_ids):
            lane = lanes[lane_id]

            nbr_ids = lane.predecessors
            if nbr_ids is not None:
                for nbr_id in nbr_ids:
                    if nbr_id in lane_ids:
                        j = lane_ids.index(nbr_id)
                        pre_pairs.append([i, j])

            nbr_ids = lane.successors
            if nbr_ids is not None:
                for nbr_id in nbr_ids:
                    if nbr_id in lane_ids:
                        j = lane_ids.index(nbr_id)
                        suc_pairs.append([i, j])

            nbr_id = lane.l_neighbor_id
            if nbr_id is not None:
                if nbr_id in lane_ids:
                    j = lane_ids.index(nbr_id)
                    left_pairs.append([i, j])

            nbr_id = lane.r_neighbor_id
            if nbr_id is not None:
                if nbr_id in lane_ids:
                    j = lane_ids.index(nbr_id)
                    right_pairs.append([i, j])
        pre_pairs = np.asarray(pre_pairs, np.int64)
        suc_pairs = np.asarray(suc_pairs, np.int64)
        left_pairs = np.asarray(left_pairs, np.int64)
        right_pairs = np.asarray(right_pairs, np.int64)

        graph = dict()
        graph['ctrs'] = np.concatenate(ctrs, 0)
        graph['num_nodes'] = num_nodes
        graph['feats'] = np.concatenate(feats, 0)
        graph['turn'] = np.concatenate(turn, 0)
        graph['control'] = np.concatenate(control, 0)
        graph['intersect'] = np.concatenate(intersect, 0)
        graph['pre'] = [pre]
        graph['suc'] = [suc]
        graph['lane_idcs'] = lane_idcs
        graph['pre_pairs'] = pre_pairs
        graph['suc_pairs'] = suc_pairs
        graph['left_pairs'] = left_pairs
        graph['right_pairs'] = right_pairs

        for k1 in ['pre', 'suc']:
            for k2 in ['u', 'v']:
                graph[k1][0][k2] = np.asarray(graph[k1][0][k2], np.int64)

        for key in ['pre', 'suc']:
            if 'scales' in self.config and self.config['scales']:
                # TODO: delete here
                graph[key] += dilated_nbrs2(graph[key][0], graph['num_nodes'], self.config['scales'])
            else:
                graph[key] += dilated_nbrs(graph[key][0], graph['num_nodes'], self.config['num_scales'])
        return graph


def get_ctrs_idx(data):
    ctrs_list = data['graph']['ctrs']
    nearest_ctrs_hist = np.zeros_like(data['feats'])[:, :, 0]
    traj = np.zeros_like(data['feats'])[:, :, :2]
    traj[:, -1, :] = data['ctrs']
    for i in range(19):
        traj[:, -2 - i, :] = traj[:, -i - 1, :] - data['feats'][:, -i - 1, :2]
    for i in range(20):
        for j in range(traj.shape[0]):
            ref_pos = traj[j, i, :]
            nearest_ctrs_hist[j, i] = np.argmin(np.linalg.norm(ctrs_list - ref_pos, axis=1))

    data['nearest_ctrs_hist'] = nearest_ctrs_hist
    return data


class ArgoTestDataset(ArgoDataset):
    def __init__(self, split, config, train=False):

        self.config = config
        self.train = train
        split2 = config['val_split'] if split == 'val' else config['test_split']
        split = self.config['preprocess_val'] if split == 'val' else self.config['preprocess_test']

        self.avl = ArgoverseForecastingLoader(split2)
        if 'preprocess' in config and config['preprocess']:
            if train:
                self.split = np.load(split, allow_pickle=True)
            else:
                self.split = np.load(split, allow_pickle=True)
        else:
            self.avl = ArgoverseForecastingLoader(split)
            self.am = ArgoverseMap()

    def __getitem__(self, idx):
        if 'preprocess' in self.config and self.config['preprocess']:
            data = self.split[idx]
            data['argo_id'] = int(self.avl.seq_list[idx].name[:-4])  # 160547

            if self.train and self.config['rot_aug']:
                # TODO: Delete Here because no rot_aug
                new_data = dict()
                for key in ['orig', 'gt_preds', 'has_preds']:
                    new_data[key] = ref_copy(data[key])

                dt = np.random.rand() * self.config['rot_size']  # np.pi * 2.0
                theta = data['theta'] + dt
                new_data['theta'] = theta
                new_data['rot'] = np.asarray([
                    [np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]], np.float32)

                rot = np.asarray([
                    [np.cos(-dt), -np.sin(-dt)],
                    [np.sin(-dt), np.cos(-dt)]], np.float32)
                new_data['feats'] = data['feats'].copy()
                new_data['feats'][:, :, :2] = np.matmul(new_data['feats'][:, :, :2], rot)
                new_data['ctrs'] = np.matmul(data['ctrs'], rot)

                graph = dict()
                for key in ['num_nodes', 'turn', 'control', 'intersect', 'pre', 'suc', 'lane_idcs', 'left_pairs', 'right_pairs']:
                    graph[key] = ref_copy(data['graph'][key])
                graph['ctrs'] = np.matmul(data['graph']['ctrs'], rot)
                graph['feats'] = np.matmul(data['graph']['feats'], rot)
                new_data['graph'] = graph
                data = new_data
            else:
                new_data = dict()
                for key in ['orig', 'gt_preds', 'has_preds', 'theta', 'rot', 'feats', 'ctrs', 'graph', 'argo_id', 'city']:
                    if key in data:
                        new_data[key] = ref_copy(data[key])
                data = new_data
            return data

        data = self.read_argo_data(idx)
        data = self.get_obj_feats(data)
        data['graph'] = self.get_lane_graph(data)
        data['idx'] = idx
        return data

    def __len__(self):
        if 'preprocess' in self.config and self.config['preprocess']:
            return len(self.split)
        else:
            return len(self.avl)


class MapQuery(object):
    # TODO: DELETE HERE No used
    """[Deprecated] Query rasterized map for a given region"""

    def __init__(self, scale, autoclip=True):
        """
        scale: one meter -> num of `scale` voxels 
        """
        super(MapQuery, self).__init__()
        assert scale in (1, 2, 4, 8)
        self.scale = scale
        root_dir = '/mnt/yyz_data_1/users/ming.liang/argo/tmp/map_npy/'
        mia_map = np.load(f"{root_dir}/mia_{scale}.npy")
        pit_map = np.load(f"{root_dir}/pit_{scale}.npy")
        self.autoclip = autoclip
        self.map = dict(
            MIA=mia_map,
            PIT=pit_map
        )
        self.OFFSET = dict(
            MIA=np.array([502, -545]),
            PIT=np.array([-642, 211]),
        )
        self.SHAPE = dict(
            MIA=(3674, 1482),
            PIT=(3043, 4259)
        )

    def query(self, region, theta=0, city='MIA'):
        """
        region: [x0,x1,y0,y1]
        city: 'MIA' or 'PIT'
        theta: rotation of counter-clockwise, angel/degree likd 90,180
        return map_mask: 2D array of shape (x1-x0)*scale, (y1-y0)*scale
        """
        region = [int(x) for x in region]

        map_data = self.map[city]
        offset = self.OFFSET[city]
        shape = self.SHAPE[city]
        x0, x1, y0, y1 = region
        x0, x1 = x0 + offset[0], x1 + offset[0]
        y0, y1 = y0 + offset[1], y1 + offset[1]
        x0, x1, y0, y1 = [round(_ * self.scale) for _ in [x0, x1, y0, y1]]
        # extend the crop region to 2x -- for rotation
        H, W = y1 - y0, x1 - x0
        x0 -= int(round(W / 2))
        y0 -= int(round(H / 2))
        x1 += int(round(W / 2))
        y1 += int(round(H / 2))
        results = np.zeros([H * 2, W * 2])
        # padding of crop -- for outlier
        xstart, ystart = 0, 0
        if self.autoclip:
            if x0 < 0:
                xstart = -x0
                x0 = 0
            if y0 < 0:
                ystart = -y0
                y0 = 0
            x1 = min(x1, shape[1] * self.scale - 1)
            y1 = min(y1, shape[0] * self.scale - 1)
        map_mask = map_data[y0:y1, x0:x1]
        _H, _W = map_mask.shape
        results[ystart:ystart + _H, xstart:xstart + _W] = map_mask
        results = results[::-1]  # flip to cartesian
        # rotate and remove margin
        rot_map = rotate(results, theta, center=None, order=0)  # center None->map center
        H, W = results.shape
        outputH, outputW = round(H / 2), round(W / 2)
        startH, startW = round(H // 4), round(W // 4)
        crop_map = rot_map[startH:startH + outputH, startW:startW + outputW]
        return crop_map


def ref_copy(data):
    if isinstance(data, list):
        return [ref_copy(x) for x in data]
    if isinstance(data, dict):
        d = dict()
        for key in data:
            d[key] = ref_copy(data[key])
        return d
    return data


def dilated_nbrs(nbr, num_nodes, num_scales):
    data = np.ones(len(nbr['u']), np.bool)
    csr = sparse.csr_matrix((data, (nbr['u'], nbr['v'])), shape=(num_nodes, num_nodes))

    mat = csr
    nbrs = []
    for i in range(1, num_scales):
        mat = mat * mat

        nbr = dict()
        coo = mat.tocoo()
        nbr['u'] = coo.row.astype(np.int64)
        nbr['v'] = coo.col.astype(np.int64)
        nbrs.append(nbr)
    return nbrs


def dilated_nbrs2(nbr, num_nodes, scales):
    data = np.ones(len(nbr['u']), np.bool)
    csr = sparse.csr_matrix((data, (nbr['u'], nbr['v'])), shape=(num_nodes, num_nodes))

    mat = csr
    nbrs = []
    for i in range(1, max(scales)):
        mat = mat * csr

        if i + 1 in scales:
            nbr = dict()
            coo = mat.tocoo()
            nbr['u'] = coo.row.astype(np.int64)
            nbr['v'] = coo.col.astype(np.int64)
            nbrs.append(nbr)
    return nbrs


def collate_fn(batch):
    batch = from_numpy(batch)
    return_batch = dict()
    # Batching by use a list for non-fixed size
    for key in batch[0].keys():
        return_batch[key] = [x[key] for x in batch]

    return return_batch


def from_numpy(data):
    """Recursively transform numpy.ndarray to torch.Tensor.
    """
    if isinstance(data, dict):
        for key in data.keys():
            data[key] = from_numpy(data[key])
    if isinstance(data, list) or isinstance(data, tuple):
        data = [from_numpy(x) for x in data]
    if isinstance(data, np.ndarray):
        """Pytorch now has bool type."""
        data = torch.from_numpy(data)
    return data


def cat(batch):
    if torch.is_tensor(batch[0]):
        batch = [x.unsqueeze(0) for x in batch]
        return_batch = torch.cat(batch, 0)
    elif isinstance(batch[0], list) or isinstance(batch[0], tuple):
        batch = zip(*batch)
        return_batch = [cat(x) for x in batch]
    elif isinstance(batch[0], dict):
        return_batch = dict()
        for key in batch[0].keys():
            return_batch[key] = cat([x[key] for x in batch])
    else:
        return_batch = batch
    return return_batch
