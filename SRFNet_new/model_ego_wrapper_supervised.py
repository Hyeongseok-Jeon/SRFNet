import numpy as np
import os
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.autograd import Variable
from SRFNet_new.utils import gpu, to_long, Optimizer, StepLR, to_float
from numpy import float64, ndarray
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union


### end of config ###
class model(nn.Module):
    def __init__(self, config, args, base_net):
        super(model, self).__init__()
        self.config = config
        self.args = args

        self.base_net = base_net
        self.ego_react_encoder = EgoReactEncodeNet(config)
        self.generator = GenerateNet(config)

    def forward(self, data):
        batch_num = len(data['gt_preds'])
        ego_fut_traj = [gpu(data['gt_preds'][i][0][0:1, :, :]) for i in range(batch_num)]

        hid = [gpu(data['action_input'][i][0]) for i in range(batch_num)]

        init_pred_global_raw = [gpu(data['init_pred_global'][i][0]) for i in range(batch_num)]
        init_pred = dict()
        init_pred['cls'] = []
        init_pred['reg'] = []
        for i in range(len(hid)):
            init_pred['cls'].append(init_pred_global_raw[i][0]['cls'][0])
            init_pred['reg'].append(init_pred_global_raw[i][0]['reg'][0])
        init_pred_global = [init_pred]

        mus_enc, _ = self.ego_react_encoder(ego_fut_traj, hid)
        delta = self.generator(mus_enc, batch_num)
        init_pred_global[0]['reg'] = [init_pred_global[0]['reg'][i] + delta[i] for i in range(batch_num)]
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

    def forward(self, mus_enc, batch_num):

        out = self.lstm(mus_enc)[0]
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
        self.pred_loss = PredLoss(config)

    def forward(self, out: Dict, data: Dict) -> Dict:
        loss_out = self.pred_loss(out, gpu(data["gt_preds"]), gpu(data["has_preds"]))
        loss_out["loss"] = loss_out["cls_loss"] / (
            loss_out["num_cls"] + 1e-10
        ) + loss_out["reg_loss"] / (loss_out["num_reg"] + 1e-10)
        return loss_out


class PostProcess(nn.Module):
    def __init__(self, config):
        super(PostProcess, self).__init__()
        self.config = config

    def forward(self, out,data):
        post_out = dict()
        post_out["preds"] = [x[0:1].detach().cpu().numpy() for x in out["reg"]]
        post_out["gt_preds"] = [x[0:1].numpy() for x in data["gt_preds"]]
        post_out["has_preds"] = [x[0:1].numpy() for x in data["has_preds"]]
        return post_out

    def append(self, metrics: Dict, loss_out: Dict, post_out: Optional[Dict[str, List[ndarray]]]=None) -> Dict:
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
