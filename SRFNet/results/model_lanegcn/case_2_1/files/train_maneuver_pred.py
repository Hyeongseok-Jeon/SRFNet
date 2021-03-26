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
import horovod.torch as hvd
from SRFNet.utils import gpu, to_long, Optimizer, StepLR
from argoverse.map_representation.map_api import ArgoverseMap

from torch.utils.data.distributed import DistributedSampler

from SRFNet.utils import Logger, load_pretrain
import math
from mpi4py import MPI

comm = MPI.COMM_WORLD
hvd.init()
torch.cuda.set_device(hvd.local_rank())

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
am = ArgoverseMap()


def main():
    seed = hvd.rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Import all settings for experiment.
    args = parser.parse_args()
    model = import_module(args.model)
    config, Dataset, collate_fn, net, loss, post_process, opt = model.get_model(args)

    config['model'] = args.model
    if config["horovod"]:
        opt.opt = hvd.DistributedOptimizer(
            opt.opt, named_parameters=net.named_parameters()
        )

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
        val_sampler = DistributedSampler(
            dataset, num_replicas=hvd.size(), rank=hvd.rank()
        )
        val_loader = DataLoader(
            dataset,
            batch_size=config["val_batch_size"],
            num_workers=config["val_workers"],
            sampler=val_sampler,
            collate_fn=collate_fn,
            pin_memory=True,
        )

        hvd.broadcast_parameters(net.state_dict(), root_rank=0)
        val(config, val_loader, net, loss, post_process, 999)
        return

    # Create log and copy all code
    save_dir = config["save_dir"]
    log = os.path.join(save_dir, "log")
    if hvd.rank() == 0:
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
    dataset = Dataset(config["train_split"], config, train=True)
    train_sampler = DistributedSampler(
        dataset, num_replicas=hvd.size(), rank=hvd.rank()
    )
    train_loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        num_workers=config["workers"],
        sampler=train_sampler,
        collate_fn=collate_fn,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        drop_last=True,
    )

    # Data loader for evaluation
    dataset = Dataset(config["val_split"], config, train=False)
    val_sampler = DistributedSampler(dataset, num_replicas=hvd.size(), rank=hvd.rank())
    val_loader = DataLoader(
        dataset,
        batch_size=config["val_batch_size"],
        num_workers=config["val_workers"],
        sampler=val_sampler,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    hvd.broadcast_parameters(net.state_dict(), root_rank=0)
    if config["horovod"]:
        hvd.broadcast_optimizer_state(opt.opt, root_rank=0)

    epoch = config["epoch"]
    remaining_epochs = int(np.ceil(config["num_epochs"] - epoch))
    for i in range(remaining_epochs):
        train(epoch + i, config, train_loader, net, loss, post_process, opt, val_loader)


def worker_init_fn(pid):
    np_seed = hvd.rank() * 1024 + int(pid)
    np.random.seed(np_seed)
    random_seed = np.random.randint(2 ** 32 - 1)
    random.seed(random_seed)


def train(epoch, config, train_loader, net, loss, post_process, opt, val_loader=None):
    train_loader.sampler.set_epoch(int(epoch))
    net.train()

    num_batches = len(train_loader)
    epoch_per_batch = 1.0 / num_batches
    save_iters = int(np.ceil(config["save_freq"] * num_batches))
    display_iters = int(
        config["display_iters"] / (hvd.size() * config["batch_size"])
    )
    val_iters = int(config["val_iters"] / (hvd.size() * config["batch_size"]))

    start_time = time.time()
    metrics = dict()
    for i, data in tqdm(enumerate(train_loader), disable=hvd.rank()):
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

        num_iters = int(np.round(epoch * num_batches))
        if hvd.rank() == 0 and (
                num_iters % save_iters == 0 or epoch >= config["num_epochs"]
        ):
            save_ckpt(net, opt, config["save_dir"], epoch)

        if num_iters % display_iters == 0:
            dt = time.time() - start_time
            metrics = sync(metrics)
            if hvd.rank() == 0:
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
    metrics = sync(metrics)
    if hvd.rank() == 0:
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


def sync(data):
    data_list = comm.allgather(data)
    data = dict()
    for key in data_list[0]:
        if isinstance(data_list[0][key], list):
            data[key] = []
        else:
            data[key] = 0
        for i in range(len(data_list)):
            data[key] += data_list[i][key]
    return data


if __name__ == "__main__":
    main()

# TODO: modify prediction head : classify and regression
# TODO: GAN wrapper (with vanila GAN, conditional GAN, info GAN)
