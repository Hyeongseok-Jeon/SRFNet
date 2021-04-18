import os
from utils import gpu, to_long, Optimizer, StepLR, to_float

def get_config(args):
    file_path = os.path.abspath(__file__)
    root_path = os.path.dirname(file_path)
    model_name = args.case
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

    config["batch_size"] = 128
    config["val_batch_size"] = 128
    config["workers"] = 12
    config["val_workers"] = config["workers"]

    """Dataset"""
    # Raw Dataset
    config["train_split"] = os.path.join(
        root_path, "dataset/train/data"
    )
    config["val_split"] = os.path.join(root_path, "dataset/val/data")

    # Preprocessed Dataset
    config["preprocess"] = True  # whether use preprocess or not
    config["preprocess_train"] = os.path.join(
        root_path, "dataset", "preprocess_GAN", "train", "train_crs_dist6_angle90.p"
    )
    config["preprocess_val"] = os.path.join(
        root_path, "dataset", "preprocess_GAN", "val", "val_crs_dist6_angle90.p"
    )
    config["SRF_data_train_dir"] = os.path.join(
        root_path, "dataset", "preprocess_GAN", "train"
    )
    config["SRF_data_val_dir"] = os.path.join(
        root_path, "dataset", "preprocess_GAN", "val"
    )
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

    return config