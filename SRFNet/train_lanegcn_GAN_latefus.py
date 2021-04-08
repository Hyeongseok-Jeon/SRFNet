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
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import horovod.torch as hvd
from torch.utils.data.distributed import DistributedSampler
from utils import Logger, load_pretrain
from mpi4py import MPI

# import SRFNet.model_lanegcn_GAN_latefus as model
comm = MPI.COMM_WORLD
hvd.init()
torch.cuda.set_device(hvd.local_rank())

root_path = os.path.dirname(os.path.abspath(__file__))
# root_path = os.getcwd()
sys.path.insert(0, root_path)

parser = argparse.ArgumentParser(description="Fuse Detection in Pytorch")
parser.add_argument(
    "-m", "--model", default="model_lanegcn_GAN_latefus", type=str, metavar="MODEL", help="model name"
)
parser.add_argument("--eval", action="store_true")
parser.add_argument(
    "--resume", default="", type=str, metavar="RESUME", help="checkpoint path"
)
parser.add_argument(
    "--weight", default="", type=str, metavar="WEIGHT", help="checkpoint path"
)
parser.add_argument(
    "--case", default="vanilla_gan", type=str
)

# parser.add_argument("--mode", default='client')
# parser.add_argument("--port", default=52162)
margin = 0.35
equilibrium = 0.68


def main():
    seed = hvd.rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Import all settings for experiment.
    args = parser.parse_args()
    model = import_module(args.model)
    config, Dataset, collate_fn, net, loss, post_process, opt, params = model.get_model(args)

    if config["horovod"]:
        for i in range(len(opt)):
            opt[i].opt = hvd.DistributedOptimizer(
                opt[i].opt, named_parameters=params[i], backward_passes_per_step=10
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
        for i in range(len(opt)):
            if opt[i] != None:
                hvd.broadcast_optimizer_state(opt[i].opt, root_rank=0)

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

    num_batches = len(train_loader)
    epoch_per_batch = 1.0 / num_batches
    save_iters = int(np.ceil(config["save_freq"] * num_batches))
    display_iters = int(
        config["display_iters"] / (hvd.size() * config["batch_size"])
    )
    val_iters = int(config["val_iters"] / (hvd.size() * config["batch_size"]))

    start_time = time.time()
    metrics = dict()

    opt_enc = opt[0]
    opt_gen = opt[1]
    opt_dis = opt[2]
    for i, data in tqdm(enumerate(train_loader), disable=hvd.rank()):
        epoch += epoch_per_batch
        data = dict(data)
        output, traj_gt, traj_pred, layer_gt, layer_pred, label_gt, label_pred, label_sample, mus, variances = get_out(net, data)
        loss_out = loss(traj_gt, traj_pred, layer_gt, layer_pred, label_gt, label_pred, label_sample, mus, variances, data, output)

        reconstruction_loss = loss_out['reconstruction_loss']
        kl_loss = loss_out['kl_loss']
        mae_hidden_loss = loss_out['mae_hidden_loss']
        bce_dis_pred = loss_out['bce_dis_pred']
        bce_dis_gt = loss_out['bce_dis_gt']

        opt_enc.zero_grad()
        loss_encoder = kl_loss + mae_hidden_loss
        loss_encoder.backward()
        lr_enc = opt_enc.step(epoch)

        train_dis = True
        train_dec = True
        if bce_dis_gt < equilibrium - margin or bce_dis_pred < equilibrium - margin:
            train_dis = False
        if bce_dis_gt > equilibrium + margin or bce_dis_pred > equilibrium + margin:
            train_dec = False
        if train_dec is False and train_dis is False:
            train_dis = True
            train_dec = True

        if train_dec:
            output, traj_gt, traj_pred, layer_gt, layer_pred, label_gt, label_pred, label_sample, mus, variances = get_out(net, data)
            loss_out = loss(traj_gt, traj_pred, layer_gt, layer_pred, label_gt, label_pred, label_sample, mus, variances, data, output)

            reconstruction_loss = loss_out['reconstruction_loss']
            kl_loss = loss_out['kl_loss']
            mae_hidden_loss = loss_out['mae_hidden_loss']
            bce_gen_sample = loss_out['bce_gen_sample']
            bce_gen_pred = loss_out['bce_gen_pred']

            opt_gen.zero_grad()
            loss_generator = 0.1 * mae_hidden_loss + (1.0 - 0.1) * (bce_gen_pred + bce_gen_sample)
            loss_generator.backward()
            lr_gen = opt_gen.step(epoch)

        if train_dis:
            output, traj_gt, traj_pred, layer_gt, layer_pred, label_gt, label_pred, label_sample, mus, variances = get_out(net, data)
            loss_out = loss(traj_gt, traj_pred, layer_gt, layer_pred, label_gt, label_pred, label_sample, mus, variances, data, output)

            reconstruction_loss = loss_out['reconstruction_loss']
            bce_dis_sample = loss_out['bce_dis_sample']
            bce_dis_gt = loss_out['bce_dis_gt']

            opt_dis.zero_grad()
            loss_discriminator = bce_dis_gt + bce_dis_sample
            loss_discriminator.backward()
            lr_gen = opt_dis.step(epoch)

        out_added = output[0]
        post_out = post_process(out_added, data)
        post_process.append(metrics, loss_out, post_out)

        num_iters = int(np.round(epoch * num_batches))
        if hvd.rank() == 0 and (
                num_iters % save_iters == 0 or epoch >= config["num_epochs"]
        ):
            save_ckpt(net, opt_enc, opt_gen, opt_dis, config["save_dir"], epoch)

        if num_iters % display_iters == 0:
            dt = time.time() - start_time
            metrics = sync(metrics)
            if hvd.rank() == 0:
                post_process.display(metrics, dt, epoch, lr_enc)
            start_time = time.time()
            metrics = dict()

        if num_iters % val_iters == 0:
            val(config, val_loader, net, loss, post_process, epoch)

        if epoch >= config["num_epochs"]:
            val(config, val_loader, net, loss, post_process, epoch)
            return


def val(config, data_loader, net, loss, post_process, epoch):
    start_time = time.time()
    metrics = dict()
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            data = dict(data)
            output, traj_gt, traj_pred, layer_gt, layer_pred, label_gt, label_pred, label_sample, mus, variances = get_out(net, data)
            loss_out = loss(traj_gt, traj_pred, layer_gt, layer_pred, label_gt, label_pred, label_sample, mus, variances, data, output)
            out_added =  output[0]
            post_out = post_process(out_added, data)
            post_process.append(metrics, loss_out, post_out)

    dt = time.time() - start_time
    metrics = sync(metrics)
    if hvd.rank() == 0:
        post_process.display(metrics, dt, epoch)


def save_ckpt(net, opt_enc, opt_gen, opt_dis, save_dir, epoch):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    state_dict = net.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].cpu()

    save_name = "%3.3f.ckpt" % epoch
    torch.save(
        {"epoch": epoch, "state_dict": state_dict, "opt_enc_state": opt_enc.opt.state_dict(), "opt_gen_state": opt_gen.opt.state_dict(), "opt_dis_state": opt_dis.opt.state_dict()},
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

def get_out(net, data):
    output, dis, dis_layer, mus, log_vars, target_gt = net(data)
    traj_gt = target_gt
    traj_pred = output[0]['reg']
    layer_gt = [dis_layer[0][i:i + 1, :] for i in range(len(traj_gt))]
    layer_pred = [dis_layer[1][6 * i:6 * (i + 1), :] for i in range(len(traj_gt))]
    label_gt = [dis[0][i] for i in range(len(traj_gt))]
    label_pred = [dis[1][6 * i:6 * (i + 1), 0] for i in range(len(traj_gt))]
    label_sample = [dis[2][6 * i:6 * (i + 1), 0] for i in range(len(traj_gt))]
    mus = [mus[i].unsqueeze(dim=0) for i in range(len(traj_gt))]
    variances = [log_vars[i].unsqueeze(dim=0) for i in range(len(traj_gt))]

    return output, traj_gt, traj_pred, layer_gt, layer_pred, label_gt, label_pred, label_sample, mus, variances

if __name__ == "__main__":
    main()