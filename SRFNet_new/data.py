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


class ArgoDataset(Dataset):
    def __init__(self, split, config, train=True):
        self.config = config
        self.train = train
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
                for key in ['city', 'orig', 'gt_preds', 'has_preds',
                            'theta', 'rot', 'feats', 'ego_feats',
                            'ctrs', 'graph', 'file_name', 'cl_cands',
                            'cl_cands_mod', 'gt_cl_cands', 'action_input',
                            'init_pred_global', 'init_pred_global_con', 'idx']:
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
        data['cl_cands'] = self.get_cl_cands(data)
        cl_cands_mod, gt_cl_cands = cl_cands_gather(data['cl_cands'], data['feats'], data)
        # data['cl_cands_mod'] = cl_cands_mod
        data['gt_cl_cands'] = gt_cl_cands

        return data

    def __len__(self):
        if 'preprocess' in self.config and self.config['preprocess']:
            return len(self.split)
        else:
            return len(self.avl)

    def get_cl_cands(self, data):
        cl_cands = []
        hist_feats = data['feats']
        hist_traj_tmp = np.zeros((hist_feats.shape[0], 20, 2))
        for j in range(19, -1, -1):
            if j == 19:
                hist_traj_tmp[:, j, :] = data['ctrs']
            else:
                hist_traj_tmp[:, j, :] = hist_traj_tmp[:, j + 1, :] - hist_feats[:, j + 1, :2]
        hist_traj_tmp = np.transpose(np.matmul(np.linalg.inv(data['rot']), np.transpose(hist_traj_tmp, (0, 2, 1))), (0, 2, 1))
        hist_traj_tmp = hist_traj_tmp + data['orig']
        for j in range(hist_feats.shape[0]):
            cl_list_mod = []
            moving_dist = np.linalg.norm(np.sum(hist_feats[j], axis=0)[:2])
            if moving_dist > 1.5 or j == 1:
                cl_list = self.am.get_candidate_centerlines_for_traj(hist_traj_tmp[j], data['city'], viz=False)
                for k in range(len(cl_list)):
                    init_idx = np.argmin(np.linalg.norm(cl_list[k] - hist_traj_tmp[j][:1, :], axis=1))
                    cl_sparse = sparse_wp(cl_list[k][init_idx:, :])
                    cl_list_mod.append(cl_sparse)
            cl_cands.append(cl_list_mod)
        return cl_cands

    def read_argo_data(self, idx):
        city = copy.deepcopy(self.avl[idx].city)

        """TIMESTAMP,TRACK_ID,OBJECT_TYPE,X,Y,CITY_NAME"""
        df = copy.deepcopy(self.avl[idx].seq_df)

        agt_ts = np.sort(np.unique(df['TIMESTAMP'].values))
        mapping = dict()
        for i, ts in enumerate(agt_ts):
            mapping[ts] = i

        trajs = np.concatenate((
            df.X.to_numpy(dtype=np.float32).reshape(-1, 1),
            df.Y.to_numpy(dtype=np.float32).reshape(-1, 1)), 1)

        steps = [mapping[x] for x in df['TIMESTAMP'].values]
        steps = np.asarray(steps, np.int64)

        objs = df.groupby(['TRACK_ID', 'OBJECT_TYPE']).groups
        keys = list(objs.keys())
        obj_type = [x[1] for x in keys]

        av_idx = obj_type.index('AV')
        idcs_ego = objs[keys[av_idx]]

        agt_idx = obj_type.index('AGENT')
        idcs = objs[keys[agt_idx]]

        av_traj = trajs[idcs_ego]
        av_step = steps[idcs_ego]
        agt_traj = trajs[idcs]
        agt_step = steps[idcs]
        file_name = self.avl[idx].current_seq
        if av_idx < agt_idx:
            del keys[av_idx]
            del keys[agt_idx - 1]
        else:
            del keys[agt_idx]
            del keys[av_idx - 1]

        ctx_trajs, ctx_steps = [], []
        for key in keys:
            idcs = objs[key]
            ctx_trajs.append(trajs[idcs])
            ctx_steps.append(steps[idcs])

        data = dict()
        data['city'] = city
        data['trajs'] = [av_traj] + [agt_traj] + ctx_trajs
        data['steps'] = [av_step] + [agt_step] + ctx_steps
        data['file_name'] = file_name
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
        for j in range(len(data['trajs'])):
            traj = data['trajs'][j]
            step = data['steps'][j]
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
            if j > 1:
                if feat[-1, 0] < x_min or feat[-1, 0] > x_max or feat[-1, 1] < y_min or feat[-1, 1] > y_max:
                    continue
            if j == 0:
                ego_traj = data['trajs'][j]
                ego_feat = np.zeros((data['trajs'][j].shape[0], 3), np.float32)
                ego_feat[:, :2] = np.matmul(rot, (ego_traj - orig.reshape(-1, 2)).T).T
                ego_feat[:, 2] = 1.0
                ego_feat[1:, :2] -= ego_feat[:-1, :2]
                ego_feat[0, :2] = 0

            ctrs.append(feat[-1, :2].copy())
            feat[1:, :2] -= feat[:-1, :2]
            feat[step[0], :2] = 0
            feats.append(feat)
            gt_preds.append(gt_pred)
            has_preds.append(has_pred)

        feats = np.asarray(feats, np.float32)
        ego_feats = np.asarray(ego_feat, np.float32)
        ctrs = np.asarray(ctrs, np.float32)
        gt_preds = np.asarray(gt_preds, np.float32)
        has_preds = np.asarray(has_preds, np.bool)

        data['feats'] = feats
        data['ego_feats'] = ego_feats
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

    for keys in return_batch.keys():
        return_batch[keys] = [return_batch[keys][i][0] for i in range(len(return_batch[keys]))]

    orig_tot = torch.cat([return_batch['orig'][i].unsqueeze(dim=0) for i in range(len(return_batch['orig']))])  # (batch num, 2)
    gt_preds_tot = torch.cat(return_batch['gt_preds'])  # (vehicle num, 30, 2)
    has_preds_tot = torch.cat(return_batch['has_preds'])  # (vehicle num, 30)
    theta_tot = torch.cat([torch.from_numpy(np.asarray([[return_batch['theta'][i]]])) for i in range(len(return_batch['theta']))])  # (batch num, 1)
    rot_tot = torch.cat([return_batch['rot'][i].unsqueeze(dim=0) for i in range(len(return_batch['rot']))])  # (batch num, 2, 2)
    feats_tot = torch.cat(return_batch['feats'])  # (vehicle num, 20, 3)
    ego_feats_tot = torch.cat([return_batch['ego_feats'][i].unsqueeze(dim=0) for i in range(len(return_batch['ego_feats']))])  # (batch num, 50, 3)
    ctrs_tot = torch.cat(return_batch['ctrs'])  # (vehicle num, 2)
    action_input_tot = torch.cat([return_batch['action_input'][i].unsqueeze(dim=0) for i in range(len(return_batch['action_input']))])  # (batch num, 6, 30, 204)
    init_pred_global_reg_tot = torch.cat([return_batch['init_pred_global'][i]['reg'][0] for i in range(len(return_batch['init_pred_global']))], dim=0)  # (vehicle num, 6, 30, 2)
    init_pred_global_cls_tot = torch.cat([return_batch['init_pred_global'][i]['cls'][0] for i in range(len(return_batch['init_pred_global']))], dim=0)  # (vehicle num, 6)
    vehicle_per_batch = torch.cat([torch.tensor(return_batch['gt_preds'][i].shape[0]).unsqueeze(dim=0) for i in range(len(return_batch['gt_preds']))], dim=0)
    vehicle_per_batch_tmp = torch.cat((torch.tensor([0.], dtype=torch.float32, device=vehicle_per_batch.device), vehicle_per_batch))
    idx = []
    for i in range(batch_num + 1):
        idx.append(int(sum(vehicle_per_batch[j + 1] for j in range(i))))
    
    batch_num = orig_tot.shape[0]
    vehicle_num = gt_preds_tot.shape[0]
    data_num = 12
    max_vehicle_in_batch = torch.max(vehicle_per_batch)
    
    mask = torch.zeros(size=(batch_num, data_num, max_vehicle_in_batch, 50, 30, 2))
    mask[:, 0, 0, :2, 0, 0] = orig_tot
    mask[:, 3, 0, :1, 0, 0] = theta_tot
    mask[:, 4, 0, :2, :2, 0] = rot_tot
    mask[:, 6, 0, :50, :3, 0] = ego_feats_tot
    mask[:, 11, 0, 0, 0, 0] = vehicle_per_batch
    
    for i in range(batch_num):
        mask[i, 1, :vehicle_per_batch[i], :30, :2, 0] = gt_preds_tot[idx[i]:idx[i+1],: ,:]
        mask[i, 2, :vehicle_per_batch[i], :30, 0, 0] = has_preds_tot[idx[i]:idx[i+1],:]
        mask[i, 5, :vehicle_per_batch[i], :20, :3, 0] = feats_tot[idx[i]:idx[i+1],: ,:]
        mask[i, 7, :vehicle_per_batch[i], :2, 0, 0] = ctrs_tot[idx[i]:idx[i+1],:]
        mask[i, 9, :vehicle_per_batch[i], :6, :30, :2] = init_pred_global_reg_tot[idx[i]:idx[i+1],:, :, :]
        mask[i, 10, :vehicle_per_batch[i], :6, 0, 0] = init_pred_global_cls_tot[idx[i]:idx[i+1],:]
        mask[i, 11, 0, 0, 0, 0] = vehicle_per_batch[i]

    return mask, action_input_tot, return_batch['graph']



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


def circle_line_intersection(p2, p1, center):
    if p2[1] > p1[1]:
        y2 = p2[1] - center[1]
        y1 = p1[1] - center[1]
        x2 = p2[0] - center[0]
        x1 = p1[0] - center[0]
    else:
        y2 = p1[1] - center[1]
        y1 = p2[1] - center[1]
        x2 = p1[0] - center[0]
        x1 = p2[0] - center[0]

    dx = x2 - x1
    dy = y2 - y1
    dr = np.sqrt(dx ** 2 + dy ** 2)
    D = x1 * y2 - x2 * y1
    cand1 = [(D * dy + dx * np.sqrt(2 ** 2 * dr ** 2 - D ** 2)) / dr ** 2 + center[0], (-D * dx + dy * np.sqrt(2 ** 2 * dr ** 2 - D ** 2)) / dr ** 2 + center[1]]
    cand2 = [(D * dy - dx * np.sqrt(2 ** 2 * dr ** 2 - D ** 2)) / dr ** 2 + center[0], (-D * dx - dy * np.sqrt(2 ** 2 * dr ** 2 - D ** 2)) / dr ** 2 + center[1]]

    if min(p2[0], p1[0]) <= cand1[0] <= max(p2[0], p1[0]):
        if min(p2[0], p1[0]) <= cand2[0] <= max(p2[0], p1[0]):
            min_idx = np.argmin([np.linalg.norm(p2 - cand1), np.linalg.norm(p2 - cand2)])
            if min_idx == 0:
                point = cand1
            elif min_idx == 1:
                point = cand2
        else:
            point = cand1
    elif min(p2[0], p1[0]) <= cand2[0] <= max(p2[0], p1[0]):
        point = cand2
    else:
        point = None
    return point


def sparse_wp(cl):
    val_index = np.unique(cl[:, 0:1], return_index=True)[1]
    cl = np.concatenate([np.expand_dims(cl[sorted(val_index), 0:1], 1), np.expand_dims(cl[sorted(val_index), 1:2], 1)], axis=1)
    cl_mod = []
    dist = []
    i = 0
    while i < cl.shape[0]:
        if i == 0:
            cl_mod.append(cl[0, :])
            i += 1
        else:
            dist.append(np.linalg.norm(cl[i, :] - cl_mod[-1]))
            if dist[-1] > 2:
                while dist[-1] > 2:
                    cl_mod.append(np.asarray(circle_line_intersection(cl[i, :], cl[i - 1, :], cl_mod[-1])))
                    dist.append(np.linalg.norm(cl[i, :] - cl_mod[-1]))
            else:
                i += 1
    cl_mod = np.asarray(cl_mod, dtype=np.float32)[:, :, 0]
    return cl_mod


def disp_to_global(actors_in_batch, data):
    mask = np.zeros_like(actors_in_batch)[:, :, :2]
    cur_pos = data['orig'] + data['ctrs']
    mask[:, -1, :] = cur_pos
    for i in range(18, -1, -1):
        mask[:, i, :] = mask[:, i + 1, :] - data['feats'][:, i + 1, :2]
    return mask


def reform(ego_fut_traj, cl_cands_target, init_pred_global):
    reformed = []
    for i in range(len(cl_cands_target)):
        if len(cl_cands_target[i]) == 1:
            cl_cand = torch.from_numpy(cl_cands_target[i][0])
        else:
            cl_cand = get_nearest_cl([torch.from_numpy(cl_cands_target[i][j]) for j in range(len(cl_cands_target[i]))], init_pred_global['reg'][i])
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


def get_nearest_cl(cl_cands_target_tmp, init_pred_global_tmp):
    dist = []
    for i in range(len(cl_cands_target_tmp)):
        dist_tmp = []
        cl_tmp = cl_cands_target_tmp[i]
        cl_tmp_dense = get_cl_dense(cl_tmp)
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


def cl_cands_gather(cl_cands, actors_in_batch, data):
    cl_mask = np.zeros(shape=(50, 4))
    cl_tot = []

    veh_in_batch = len(cl_cands)
    veh_feats = disp_to_global(actors_in_batch, data)
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
