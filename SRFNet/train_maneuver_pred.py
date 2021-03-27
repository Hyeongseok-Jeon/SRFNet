# Copyright (c) 2020 Uber Technologies, Inc.
# See the License for the specific language governing permissions and
# limitations under the License.
import sys
import os

sys.path.extend(['/home/jhs/Desktop/SRFNet'])
sys.path.extend(['/home/jhs/Desktop/SRFNet/LaneGCN'])
sys.path.extend(['/home/user/Desktop/SRFNet'])
sys.path.extend(['/home/user/Desktop/SRFNet/LaneGCN'])
sys.path.extend(['/home/user/data/HyeongseokJeon/infogan_pred/SRFNet'])
sys.path.extend(['/home/user/data/HyeongseokJeon/infogan_pred/SRFNet/LaneGCN'])

os.umask(0)
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import numpy as np
import random
import sys
import time
import shutil
from importlib import import_module
from numbers import Number
from shapely.geometry import LineString, Point
from tqdm import tqdm
import torch
from torch.utils.data import Sampler, DataLoader


from SRFNet.utils import Logger, load_pretrain

root_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_path)

parser = argparse.ArgumentParser(description="Fuse Detection in Pytorch")
parser.add_argument(
    "-m", "--model", default="model_maneuver_pred", type=str, metavar="MODEL", help="model name"
)
parser.add_argument("--eval", action="store_true")
parser.add_argument(
    "--resume", default="", type=str, metavar="RESUME", help="checkpoint path"
)
parser.add_argument(
    "--weight", default="", type=str, metavar="WEIGHT", help="checkpoint path"
)
parser.add_argument(
    "--case", default="maneuver_pred", type=str
)

def main():
    # Import all settings for experiment.
    args = parser.parse_args()
    model = import_module(args.model)
    config, Dataset, collate_fn, net, loss, post_process, opt= model.get_model(args)
    config['model'] = args.model

    if args.resume or args.weight:
        ckpt_path = args.resume or args.weight
        if not os.path.isabs(ckpt_path):
            ckpt_path = os.path.join(config["save_dir"], ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
        load_pretrain(net, ckpt["state_dict"])
        if args.resume:
            config["epoch"] = ckpt["epoch"]
            opt.load_state_dict(ckpt["opt_state"])

    if args.eval:
        # Data loader for evaluation
        dataset = Dataset(config["val_split"], config, train=False)
        val_loader = DataLoader(
            dataset,
            batch_size=config["val_batch_size"],
            num_workers=config["val_workers"],
            shuffle=True,
            collate_fn=collate_fn
        )

        val(config, val_loader, net, loss, post_process, 999)
        return

    # Create log and copy all code
    save_dir = config["save_dir"]
    log = os.path.join(save_dir, "log")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    sys.stdout = Logger(log)

    src_dirs = [root_path]
    dst_dirs = [os.path.join(save_dir, "files")]
    for src_dir, dst_dir in zip(src_dirs, dst_dirs):
        files = [f for f in os.listdir(src_dir) if f.endswith(".py")]
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        for f in files:
            shutil.copy(os.path.join(src_dir, f), os.path.join(dst_dir, f))

    # Data loader for training
    dataset = Dataset(config["train_split"], config, train=False)
    train_loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        num_workers=config["workers"],
        shuffle=True,
        collate_fn=collate_fn
    )

    # Data loader for evaluation
    dataset = Dataset(config["val_split"], config, train=False)
    val_loader = DataLoader(
        dataset,
        batch_size=config["val_batch_size"],
        num_workers=config["val_workers"],
        shuffle=True,
        collate_fn=collate_fn
    )

    epoch = config["num_epochs"]
    for i in range(epoch):
        train(model, i, config, train_loader, net, loss, post_process, opt, val_loader)


def train(model, epoch, config, train_loader, net, loss, post_process, opt, val_loader=None):
    net.train()

    num_batches = len(train_loader)
    epoch_per_batch = 1.0 / num_batches
    save_iters = int(np.ceil(config["save_freq"] * num_batches))
    display_iters = int(
        config["display_iters"] / (config["batch_size"])
    )
    val_iters = int(config["val_iters"] / (config["batch_size"]))

    start_time = time.time()
    metrics = dict()
    acc = 0
    count = 0
    loss_tt = 0
    for i, data in enumerate(train_loader):
        current = (i + 1)
        percent = float(current) * 100 / num_batches
        arrow = '-' * int(percent / 100 * 20 - 1) + '>'
        spaces = ' ' * (20 - len(arrow))
        if i == 0:
            sys.stdout.write('\n' + ' %d th Epoch Progress: [%s%s] %d %%' % (epoch + 1, arrow, spaces, percent))
        else:
            sys.stdout.write('\r' + '%d th Epoch Progress: [%s%s] %d %%  [loss = %s] [acc = %s]' % (epoch + 1, arrow, spaces, percent, acc/count, loss_tt/count))

        epoch += epoch_per_batch
        gt = data['gt_cl_cands']
        data = dict(data)
        gt_mod = []
        for j in range(len(gt)):
            for k in range(len(gt[j])):
                gt_mod.append(gt[j][k])
        output, target_idcs = net(data)

        loss_tot, loss_calc_num, val_idx, pred_idx = loss(output, gt_mod)
        post_out = post_process(output, target_idcs, data)
        post_process.append(metrics, loss_tot, loss_calc_num, post_out)

        opt.zero_grad()
        loss_tot.backward()
        lr = opt.step(epoch)

        loss_tt += loss_tot.item()
        count += loss_calc_num
        acc += model.pred_metrics(post_out["out"], post_out["gt_preds"])

        num_iters = int(np.round(epoch * num_batches))
        if num_iters % save_iters == 0 or epoch >= config["num_epochs"]:
            save_ckpt(net, opt, config["save_dir"], epoch)

        if num_iters % display_iters == 0:
            dt = time.time() - start_time
            post_process.display(metrics, dt, epoch, lr)
            start_time = time.time()
            metrics = dict()

        if num_iters % val_iters == 0:
            val(config, val_loader, net, loss, post_process, epoch)

        if epoch >= config["num_epochs"]:
            val(config, val_loader, net, loss, post_process, epoch)
            return


def val(config, data_loader, net, loss, post_process, epoch):
    net.eval()

    start_time = time.time()
    metrics = dict()
    for i, data in enumerate(data_loader):
        data = dict(data)
        gt = data['gt_cl_cands']
        gt_mod = []
        for j in range(len(gt)):
            for k in range(len(gt[j])):
                gt_mod.append(gt[j][k])

        with torch.no_grad():
            output, target_idcs = net(data)
            loss_tot, loss_calc_num, val_idx, pred_idx = loss(output, gt_mod)
            post_out = post_process(output, target_idcs, data)
            post_process.append(metrics, loss_tot, loss_calc_num, post_out)
            loss_tot.backward()
            post_out = post_process(output, target_idcs, data)
            post_process.append(metrics, loss_tot, loss_calc_num, post_out)

    dt = time.time() - start_time
    post_process.display(metrics, dt, epoch)
    net.train()


def save_ckpt(net, opt, save_dir, epoch):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    state_dict = net.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].cpu()

    save_name = "%3.3f.ckpt" % epoch
    if len(opt) == 1:
        torch.save(
            {"epoch": epoch, "state_dict": state_dict, "opt_state": opt[0].opt.state_dict()},
            os.path.join(save_dir, save_name),
        )
    elif len(opt) == 2:
        torch.save(
            {"epoch": epoch, "state_dict": state_dict, "opt1_state": opt[0].opt.state_dict(), "opt2_state": opt[1].opt.state_dict()},
            os.path.join(save_dir, save_name),
        )


if __name__ == "__main__":
    main()

# TODO: modify prediction head : classify and regression
# TODO: GAN wrapper (with vanila GAN, conditional GAN, info GAN)
