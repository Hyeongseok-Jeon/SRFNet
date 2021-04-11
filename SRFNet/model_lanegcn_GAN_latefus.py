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

from SRFNet.data import ArgoDataset, collate_fn
from SRFNet.utils import gpu, to_long, Optimizer, StepLR, to_float

from SRFNet.layers import Conv1d, Res1d, Linear, LinearRes, Null, GraphAttentionLayer, GraphAttentionLayer_time_serial, GAT_SRF
from SRFNet.model_maneuver_pred import get_model as get_manuever_model
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

config["batch_size"] = 4
config["val_batch_size"] = 4
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
config["inter_dist_thres"] = 10
config['gan_noise_dim'] = 128


### end of config ###
class lanegcn_vanilla_gan_latefus(nn.Module):
    def __init__(self, config, args):
        super(lanegcn_vanilla_gan_latefus, self).__init__()
        self.config = config
        self.args = args
        self.args.case = 'lanegcn'
        _, _, _, maneuver_pred_net, _, _, _ = get_manuever_model(args)

        pre_trained_weight = torch.load(os.path.join(root_path, "results/model_lanegcn/lanegcn") + '/32.000.ckpt')
        pretrained_dict = pre_trained_weight['state_dict']
        new_model_dict = maneuver_pred_net.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in new_model_dict}
        new_model_dict.update(pretrained_dict)
        maneuver_pred_net.load_state_dict(new_model_dict)

        self.maneu_pred = maneuver_pred_net

        self.ego_react_encoder = EgoReactEncodeNet(config)
        self.generator = GenerateNet(config)
        self.discriminator = DiscriminateNet(config)

    def forward(self, data):
        cl_cands = to_float(gpu(data['cl_cands']))
        cl_cands_target = [to_float(cl_cands[i][1]) for i in range(len(cl_cands))]
        target_gt_traj = [gpu(data['gt_preds'][i][1:2, :, :]) for i in range(len(data['gt_preds']))]
        ego_fut_traj = [gpu(data['gt_preds'][i][0:1, :, :]) for i in range(len(data['gt_preds']))]
        target_hist_traj = feat_to_global(gpu(data['feats']), gpu(data['rot']), gpu(data['orig']), gpu(data['ctrs']))

        hid = gpu(data['data'])
        init_pred_global_raw = gpu(data['init_pred_global'])
        init_pred = dict()
        init_pred['cls'] = []
        init_pred['reg'] = []
        for i in range(len(hid)):
            init_pred['cls'].append(init_pred_global_raw[i][0]['cls'][0])
            init_pred['reg'].append(init_pred_global_raw[i][0]['reg'][0])
        init_pred_global = [init_pred]

        init_pred_global_con_raw = gpu(data['init_pred_global_con'])
        init_pred_con = dict()
        init_pred_con['cls'] = []
        init_pred_con['reg'] = []
        for i in range(len(hid)):
            init_pred_con['cls'].append(init_pred_global_con_raw[i]['cls'][0])
            init_pred_con['reg'].append(init_pred_global_con_raw[i]['reg'][0])
        init_pred_global_con = init_pred_con


        mus_enc, log_vars = self.ego_react_encoder(ego_fut_traj, hid)
        vars = [torch.exp(log_vars[i] * 0.5) for i in range(len(log_vars))]
        mus = torch.cat(mus_enc, dim=1)
        vars = torch.cat(vars, dim=1)

        noise = Variable(torch.randn(mus.shape).cuda(), requires_grad=True)
        noise_vae = noise * vars + mus

        delta = self.generator(noise_vae)
        delta = [torch.transpose(delta, 0, 1)[6 * i: 6 * (i + 1), :, :].unsqueeze(dim=0) for i in range(len(ego_fut_traj))]

        output_pred = init_pred_global
        output_pred[0]['reg'] = [output_pred[0]['reg'][i] + delta[i] for i in range(len(ego_fut_traj))]

        tot_traj_real = [torch.transpose(torch.cat([target_hist_traj[i], target_gt_traj[i][0]], dim=0).unsqueeze(dim=0), 1, 2) for i in range(len(cl_cands))]
        tot_traj_pred = [torch.transpose(torch.cat([torch.repeat_interleave(target_hist_traj[i].unsqueeze(dim=0), 6, dim=0), output_pred[0]['reg'][i][0]], dim=1), 1, 2) for i in range(len(cl_cands))]
        dis_real, dis_layer_real = self.discriminator(tot_traj_real, True)
        dis_pred, dis_layer_pred = self.discriminator(tot_traj_pred, True)

        noise_vae = Variable(torch.randn(mus.shape).cuda(), requires_grad=True)
        delta = self.generator(noise_vae)
        delta = [torch.transpose(delta, 0, 1)[6 * i: 6 * (i + 1), :, :].unsqueeze(dim=0) for i in range(len(ego_fut_traj))]

        output_sample = init_pred_global_con
        output_sample['reg'] = [output_sample['reg'][i] + delta[i] for i in range(len(ego_fut_traj))]

        tot_traj_pred = [torch.transpose(torch.cat([torch.repeat_interleave(target_hist_traj[i].unsqueeze(dim=0), 6, dim=0), output_sample['reg'][i][0]], dim=1), 1, 2) for i in range(len(cl_cands))]
        dis_sample = self.discriminator(tot_traj_pred, False)

        return output_pred, [dis_real, dis_pred, dis_sample], [dis_layer_real, dis_layer_pred], mus_enc, log_vars, target_gt_traj


def feat_to_global(targets, rot, orig, ctrs):
    batch_num = len(targets)
    targets_mod = [torch.zeros_like(targets[i])[0, :, :2] for i in range(batch_num)]
    for i in range(batch_num):
        target_cur_pos = ctrs[i][1]
        targets_mod[i][-1, :] = target_cur_pos
        target_disp = targets[i][1, :, :2]
        for j in range(18, -1, -1):
            targets_mod[i][j, :] = targets_mod[i][j + 1, :] - target_disp[j + 1, :]
        targets_mod[i] = (torch.matmul(torch.inverse(rot[i]), targets_mod[i].T).T + orig[i].reshape(-1, 2))

    return targets_mod


class EgoReactEncodeNet(nn.Module):
    def __init__(self, config):
        super(EgoReactEncodeNet, self).__init__()
        self.config = config
        self.enc1 = nn.Linear(204, config['n_actor'])
        self.enc2 = nn.Linear(config['n_actor'], config['n_actor'])
        self.mu_gen = nn.Linear(config['n_actor'], config['gan_noise_dim'])
        self.log_varience_gen = nn.Linear(config['n_actor'], config['gan_noise_dim'])

    def forward(self, ego_fut_traj, hid):
        data = hid
        data_cat = torch.cat([to_float(data[i][0]).view(-1, 204) for i in range(len(data))], dim=0)

        hid = F.relu(self.enc1(data_cat))
        hid = F.relu(self.enc2(hid))
        mus = self.mu_gen(hid)
        mus = mus.view(-1, 6, config['n_actor'])
        log_vars = self.log_varience_gen(hid)
        log_vars = log_vars.view(-1, 6, config['n_actor'])

        mus = [mus[30 * i:30 * (i + 1), :, :] for i in range(len(ego_fut_traj))]
        log_vars = [log_vars[30 * i:30 * (i + 1), :, :] for i in range(len(ego_fut_traj))]

        return [mus, log_vars]

    def reform(self, ego_fut_traj, cl_cands_target, init_pred_global):
        reformed = []
        for i in range(len(cl_cands_target)):
            if len(cl_cands_target[i]) == 1:
                cl_cand = cl_cands_target[i][0]
            else:
                cl_cand = self.get_nearest_cl(cl_cands_target[i], init_pred_global['reg'][i])
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
                sur_feat_1[jj][np.arange(30), sur_idx[jj]-min_idx] = 1
            sur_feat = torch.cat([torch.cat([sur_feat_1[i], sur_disp[i]], dim=1).unsqueeze(dim=0) for i in range(6)], dim=0)

            ego_disp = cl_cand[ego_idx] - ego_pos
            ego_feat_1 = mask.clone()
            ego_feat_1[np.arange(30),ego_idx-min_idx] = 1
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


class GenerateNet(nn.Module):
    def __init__(self, config):
        super(GenerateNet, self).__init__()
        self.config = config

        self.lstm = nn.LSTM(input_size=config['n_actor'],
                            hidden_size=config['n_actor'],
                            num_layers=2,
                            bidirectional=True)
        self.x_gen = nn.Linear(2 * config['n_actor'], 1)
        self.y_gen = nn.Linear(2 * config['n_actor'], 1)

    def forward(self, noise_vae):
        out = self.lstm(noise_vae)[0]
        x_out = self.x_gen(out)
        y_out = self.y_gen(out)

        return torch.cat([x_out, y_out], dim=-1)


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
            if i == len(in_channel) - 1:
                net = nn.Conv1d(in_channels=in_channel[i],
                                out_channels=out_channel[i],
                                kernel_size=kernel_size[i],
                                stride=stride[i],
                                padding=padding,
                                dilation=dilation)
                conv1d.append(net)
            else:
                net = nn.Conv1d(in_channels=in_channel[i],
                                out_channels=out_channel[i],
                                kernel_size=kernel_size[i],
                                stride=stride[i],
                                padding=padding,
                                dilation=dilation)
                conv1d.append(net)
                conv1d.append(nn.ReLU())
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
        outs = self.sigmoid(self.out(self.relu6(hid)))
        if get_hidden:
            return outs, hid
        else:
            return outs


class Loss(nn.Module):
    def __init__(self, config):
        super(Loss, self).__init__()
        self.config = config
        self.pred_loss = PredLoss(config)
        self.bceloss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, traj_gt, traj_pred, layer_gt, layer_pred, label_gt, label_pred, label_sample, mus, variances, data, output):
        batch_size = len(traj_gt)
        traj_gt = torch.cat(traj_gt, dim=0)
        traj_pred = torch.cat(traj_pred, dim=0)
        layer_gt = torch.cat(layer_gt, dim=0)
        layer_pred = torch.cat(layer_pred, dim=0)
        label_gt = torch.cat(label_gt, dim=0)
        label_pred = torch.cat(label_pred, dim=0)
        label_sample = torch.cat(label_sample, dim=0)
        mus = torch.cat(mus, dim=0)
        variances = torch.cat(variances, dim=0)

        nle_tot = []
        kl_tot = []
        mae_tot = []
        for j in range(batch_size):
            nle = [0.5 * (traj_gt[j].view(-1) - traj_pred[j, i, :, :].view(-1)) ** 2 for i in range(6)]
            kl = [-0.5 * (-variances[j, :, i, :].exp() - torch.pow(mus[j, :, i, :], 2) + variances[j, :, i, :] + 1) for i in range(6)]
            mae = [torch.abs(layer_gt[j, :] - layer_pred[j + i, :]) for i in range(6)]

            nle_tot = nle_tot + nle
            kl_tot = kl_tot + kl
            mae_tot = mae_tot + mae

        bce_dis_gt = torch.mean(self.bceloss(label_gt, Variable(torch.ones_like(label_gt.data).cuda(), requires_grad=False)))
        bce_dis_pred = torch.mean(self.bceloss(label_pred, Variable(torch.zeros_like(label_pred.data).cuda(), requires_grad=False)))
        bce_dis_sample = torch.mean(self.bceloss(label_sample, Variable(torch.zeros_like(label_sample.data).cuda(), requires_grad=False)))
        bce_gen_pred = torch.mean(self.bceloss(label_pred, Variable(torch.ones_like(label_pred.data).cuda(), requires_grad=False)))
        bce_gen_sample = torch.mean(self.bceloss(label_sample, Variable(torch.ones_like(label_sample.data).cuda(), requires_grad=False)))

        reconstruction_loss = torch.sum(sum(nle_tot)) / (len(nle_tot) * 60)
        kl_loss = torch.sum(sum(kl_tot)) / (len(kl_tot) * kl_tot[0].shape[0] * kl_tot[0].shape[1])
        mae_hidden_loss = 10000 * torch.sum(sum(mae_tot)) / (len(mae_tot) * mae_tot[0].shape[0])

        gt_preds = [gpu(data["gt_preds"])[i][1:2, :, :] for i in range(len(gpu(data["gt_preds"])))]
        has_preds = [gpu(data["has_preds"])[i][1:2, :] for i in range(len(gpu(data["has_preds"])))]
        loss_out = self.pred_loss(output, gt_preds, has_preds)
        loss_out["loss"] = loss_out["cls_loss"] / (
                loss_out["num_cls"] + 1e-10
        ) + loss_out["reg_loss"] / (loss_out["num_reg"] + 1e-10)

        loss_out['reconstruction_loss'] = reconstruction_loss
        loss_out['kl_loss'] = kl_loss
        loss_out['mae_hidden_loss'] = mae_hidden_loss
        loss_out['bce_gen_sample'] = bce_gen_sample
        loss_out['bce_gen_pred'] = bce_gen_pred
        loss_out['bce_dis_sample'] = bce_dis_sample
        loss_out['bce_dis_pred'] = bce_dis_pred
        loss_out['bce_dis_gt'] = bce_dis_gt

        return loss_out


class PredLoss(nn.Module):
    def __init__(self, config):
        super(PredLoss, self).__init__()
        self.config = config
        self.reg_loss = nn.SmoothL1Loss(reduction="mean")

    def forward(self, out: Dict[str, List[Tensor]], gt_preds: List[Tensor], has_preds: List[Tensor]) -> Dict[str, Union[Tensor, int]]:
        batch_num = len(gt_preds)
        cls, reg = out[0]["cls"], out[0]["reg"]
        cls = torch.cat([cls[i][1:2, :] for i in range(len(cls))], 0)
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

        row_idcs = torch.arange(batch_num).long().to(last_idcs.device)
        dist = []
        for j in range(num_mods):
            dist.append(
                torch.sqrt(
                    (
                            (reg[row_idcs, j, 29] - gt_preds[row_idcs, 29])
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
        num_added = len(metrics['preds'])
        cls_loss = metrics["cls_loss"] / num_added
        reg_loss = metrics["reg_loss"] / num_added

        reconstruction_loss = metrics['reconstruction_loss'] / num_added
        kl_loss = metrics['kl_loss'] / num_added
        mae_hidden_loss = metrics['mae_hidden_loss'] / num_added
        bce_gen_sample = metrics['bce_gen_sample'] / num_added
        bce_gen_pred = metrics['bce_gen_pred'] / num_added
        bce_dis_sample = metrics['bce_dis_sample'] / num_added
        bce_dis_pred = metrics['bce_dis_pred'] / num_added
        bce_dis_gt = metrics['bce_dis_gt'] / num_added

        loss_encoder = kl_loss + mae_hidden_loss
        loss_discriminator = bce_dis_gt + bce_dis_sample
        loss_generator = 0.1 * mae_hidden_loss + (1.0 - 0.1) * (bce_gen_pred + bce_gen_sample)

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
    net = lanegcn_vanilla_gan_latefus(config, args)
    params_enc = [(name, param) for name, param in net.ego_react_encoder.named_parameters()]
    params_gen = [(name, param) for name, param in net.generator.named_parameters()]
    params_dis = [(name, param) for name, param in net.discriminator.named_parameters()]

    param_enc = [p for n, p in params_enc]
    param_gen = [p for n, p in params_gen]
    param_dis = [p for n, p in params_dis]
    opt = [Optimizer(param_enc, config), Optimizer(param_gen, config), Optimizer(param_dis, config)]
    loss = Loss(config).cuda()
    params = [params_enc, params_gen, params_dis]

    net = net.cuda()
    post_process = PostProcess(config, args).cuda()
    config["save_dir"] = os.path.join(
        config["save_dir"], args.case
    )
    return config, ArgoDataset, collate_fn, net, loss, post_process, opt, params
