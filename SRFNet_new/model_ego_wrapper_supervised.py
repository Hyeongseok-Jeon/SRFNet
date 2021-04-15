import numpy as np
import os
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.autograd import Variable
from utils import gpu, to_long, Optimizer, StepLR, to_float
from numpy import float64, ndarray
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union


### end of config ###
class model(nn.Module):
    def __init__(self, config, args, base_net):
        super(model, self).__init__()
        self.config = config
        self.args = args

        self.base_net = base_net
        self.ego_react_encoder = EgoReactEncodeNet(config).cuda()
        self.generator = GenerateNet(config).cuda()

    def forward(self, data, actors):
        batch_num = len(data['gt_preds'])
        ego_fut_traj = [gpu(data['gt_preds'][i][0:1, :, :]) for i in range(batch_num)]

        hid = [gpu(data['action_input'][i]) for i in range(batch_num)]

        init_pred_global_raw = [gpu(data['init_pred_global'][i]) for i in range(batch_num)]
        init_pred = dict()
        init_pred['cls'] = []
        init_pred['reg'] = []
        for i in range(len(hid)):
            init_pred['cls'].append(init_pred_global_raw[i]['cls'][0])
            init_pred['reg'].append(init_pred_global_raw[i]['reg'][0])
        init_pred_global = [init_pred]

        mus_enc, _ = self.ego_react_encoder(ego_fut_traj, hid)
        delta = self.generator(mus_enc, actors, batch_num)
        init_pred_global[0]['cls'] = [init_pred_global[0]['cls'][i][1:2, :] for i in range(batch_num)]
        init_pred_global[0]['reg'] = [init_pred_global[0]['reg'][i][1, :, :, :] + delta[i] for i in range(batch_num)]
        output_pred = init_pred_global

        return output_pred


class EgoReactEncodeNet(nn.Module):
    def __init__(self, config):
        super(EgoReactEncodeNet, self).__init__()
        self.config = config
        self.enc1 = nn.Linear(204, config['n_actor'])
        self.enc2 = nn.Linear(config['n_actor'], config['n_actor'])
        self.mu_gen = nn.Linear(config['n_actor'], config['gan_noise_dim'])
        self.log_varience_gen = nn.Linear(config['n_actor'], config['gan_noise_dim'])

    def forward(self, ego_fut_traj, hid):
        datas = hid
        data_cat = torch.cat([to_float(datas[i]).view(-1, 204) for i in range(len(datas))], dim=0)

        hid = F.relu(self.enc1(data_cat))
        hid = F.relu(self.enc2(hid))
        mus = self.mu_gen(hid)
        mus = mus.view(-1, 6, self.config['n_actor'])
        log_vars = self.log_varience_gen(hid)
        log_vars = log_vars.view(-1, 6, self.config['n_actor'])

        mus = [mus[30 * i:30 * (i + 1), :, :] for i in range(len(ego_fut_traj))]
        log_vars = [log_vars[30 * i:30 * (i + 1), :, :] for i in range(len(ego_fut_traj))]

        return [mus, log_vars]


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

    def forward(self, mus_enc, actors, batch_num):
        actor = actors[0]
        actor_idcs = actors[1]

        mus_in = torch.cat(mus_enc, dim=1)
        actor_target = torch.cat([torch.repeat_interleave(actor[actor_idcs[i][1]:actor_idcs[i][1] + 1], 6, dim=0) for i in range(len(actor_idcs))], dim=0)
        actor_in = torch.repeat_interleave(actor_target.unsqueeze(dim=0), 4, dim=0)
        out = self.lstm(mus_in, (torch.zeros_like(actor_in), actor_in))[0]
        x_out = self.x_gen(out)
        y_out = self.y_gen(out)
        delta = torch.cat([x_out, y_out], dim=-1)
        delta = [torch.transpose(delta, 0, 1)[6 * i: 6 * (i + 1), :, :].unsqueeze(dim=0) for i in range(batch_num)]

        return delta


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
        self.pred_loss = nn.L1Loss(reduction='none')
        self.lanegcn_loss = PredLoss(config)

    def forward(self, out: Dict, data: Dict) -> Dict:
        gt = gpu([data["gt_preds"][i][1:2, :, :] for i in range(len(data["gt_preds"]))])
        has = gpu([data["has_preds"][i][1:2, :] for i in range(len(data["has_preds"]))])
        preds = [out['reg'][i][0] for i in range(len(data['gt_preds']))]
        preds_cls = [out['cls'][i][0] for i in range(len(data['gt_preds']))]

        loss_out = self.lanegcn_loss(out, gt, has)
        loss_out["loss"] = loss_out["cls_loss"] / (
                loss_out["num_cls"] + 1e-10
        ) + loss_out["reg_loss"] / (loss_out["num_reg"] + 1e-10)

        heading = []
        for i in range(len(gt)):
            head = torch.zeros_like(gt[0][:, :, 0])
            if torch.norm(gt[i][0, 0, :] - gt[i][0, -1, :]) > 2:
                head[0, 0] = torch.rad2deg(torch.atan2(gt[i][0, 1, 1] - gt[i][0, 0, 1], gt[i][0, 1, 0] - gt[i][0, 0, 0]))
                head[0, -1] = torch.rad2deg(torch.atan2(gt[i][0, -1, 1] - gt[i][0, -2, 1], gt[i][0, -1, 0] - gt[i][0, -2, 0]))
                zero_idx = torch.cat(
                    torch.where((torch.atan2(gt[i][0, 2:, 1] - gt[i][0, 1:-1, 1], gt[i][0, 2:, 0] - gt[i][0, 1:-1, 0])) == 0) + torch.where((torch.atan2(gt[i][0, 1:-1, 1] - gt[i][0, :-2, 1], gt[i][0, 1:-1, 0] - gt[i][0, 0:-2, 0])) == 0))
                head_tmp = torch.rad2deg(torch.atan2(gt[i][0, 2:, 1] - gt[i][0, 1:-1, 1], gt[i][0, 2:, 0] - gt[i][0, 1:-1, 0])) + torch.rad2deg(torch.atan2(gt[i][0, 1:-1, 1] - gt[i][0, :-2, 1], gt[i][0, 1:-1, 0] - gt[i][0, 0:-2, 0]))
                head[0, 1:-1] = head_tmp / 2
                head[0, zero_idx + 1] = head_tmp[zero_idx]
            heading.append(head)

        dist_error = []
        top_1_idx = []
        for i in range(len(gt)):
            gt_for_loss = torch.repeat_interleave(gt[i], 6, dim=0)
            pred_for_loss = preds[i]
            dist_error_init = self.pred_loss(gt_for_loss, pred_for_loss)

            for ii in range(30):
                rot = torch.zeros((2, 2), device="cuda")
                rot[0,0] = torch.cos(torch.deg2rad(-heading[i][0, ii]))
                rot[0,1] = -torch.sin(torch.deg2rad(-heading[i][0, ii]))
                rot[1,0] = torch.sin(torch.deg2rad(-heading[i][0, ii]))
                rot[1,1] = torch.cos(torch.deg2rad(-heading[i][0, ii]))
                dist_error_init[:,ii,:] = torch.matmul(rot, dist_error_init[:,ii,:].T).T

            dist_error.append(torch.abs(dist_error_init))
            top_1_idx.append(torch.argmax(preds_cls[i]) + 6 * i)

        dist_error = torch.cat(dist_error, dim=0)
        ade6_x_sum = torch.sum(dist_error[:, :, 0])
        ade6_y_sum = torch.sum(dist_error[:, :, 1])
        fde6_x_sum = torch.sum(dist_error[:, -1, 0])
        fde6_y_sum = torch.sum(dist_error[:, -1, 1])
        ade6_num = dist_error.shape[0] * dist_error.shape[1]
        fde6_num = dist_error.shape[0]

        ade1_x_sum = torch.sum(dist_error[top_1_idx, :, 0])
        ade1_y_sum = torch.sum(dist_error[top_1_idx, :, 1])
        fde1_x_sum = torch.sum(dist_error[top_1_idx, -1, 0])
        fde1_y_sum = torch.sum(dist_error[top_1_idx, -1, 1])
        ade1_num = len(top_1_idx) * dist_error.shape[1]
        fde1_num = len(top_1_idx)

        loss_out["ade6_x_sum"] = ade6_x_sum
        loss_out["ade6_y_sum"] = ade6_y_sum
        loss_out["fde6_x_sum"] = fde6_x_sum
        loss_out["fde6_y_sum"] = fde6_y_sum
        loss_out["ade6_num"] = ade6_num
        loss_out["fde6_num"] = fde6_num
        loss_out["ade1_x_sum"] = ade1_x_sum
        loss_out["ade1_y_sum"] = ade1_y_sum
        loss_out["fde1_x_sum"] = fde1_x_sum
        loss_out["fde1_y_sum"] = fde1_y_sum
        loss_out["ade1_num"] = ade1_num
        loss_out["fde1_num"] = fde1_num

        return loss_out


class PostProcess(nn.Module):
    def __init__(self, config):
        super(PostProcess, self).__init__()
        self.config = config

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

        ade1_x = metrics["ade1_x_sum"] / metrics["ade1_num"]
        ade1_y = metrics["ade1_y_sum"] / metrics["ade1_num"]
        ade6_x = metrics["ade6_x_sum"] / metrics["ade6_num"]
        ade6_y = metrics["ade6_y_sum"] / metrics["ade6_num"]
        fde1_x = metrics["fde1_x_sum"] / metrics["fde1_num"]
        fde1_y = metrics["fde1_y_sum"] / metrics["fde1_num"]
        fde6_x = metrics["fde6_x_sum"] / metrics["fde6_num"]
        fde6_y = metrics["fde6_y_sum"] / metrics["fde6_num"]

        cls = metrics["cls_loss"] / (metrics["num_cls"] + 1e-10)
        reg = metrics["reg_loss"] / (metrics["num_reg"] + 1e-10)
        loss = cls + reg

        preds = np.concatenate(metrics["preds"], 0)
        gt_preds = np.concatenate(metrics["gt_preds"], 0)
        has_preds = np.concatenate(metrics["has_preds"], 0)
        ade1, fde1, ade6, fde6, min_idcs = pred_metrics(preds, gt_preds, has_preds)

        print(
            "loss %2.4f %2.4f %2.4f, ade1 %2.4f, ade1_x %2.4f, ade1_y %2.4f, fde1 %2.4f, fde1_x %2.4f, fde1_y %2.4f, ade6 %2.4f, ade6_x %2.4f, ade6_y %2.4f, fde6 %2.4f, fde6_x %2.4f, fde6_y %2.4f"
            % (loss, cls, reg, ade1, ade1_x, ade1_y, fde1, fde1_x, fde1_y, ade6, ade6_x, ade6_y, fde6, fde6_x, fde6_y)
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
