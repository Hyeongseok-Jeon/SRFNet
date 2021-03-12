import os
from LaneGCN.utils import gpu, to_long,  Optimizer, StepLR

def get_config(root_path):
    config = dict()
    # Preprocessed Dataset
    config['preprocess'] = True
    config["preprocess_train"] = os.path.join(root_path, "SRFNet", "dataset", "preprocess", "train_crs_dist6_angle90.p")
    config["preprocess_val"] = os.path.join(root_path, "SRFNet", "dataset", "preprocess", "val_crs_dist6_angle90.p")
    config['preprocess_test'] = os.path.join(root_path, "SRFNet", "dataset", 'preprocess', 'test_test.p')

    # Raw Dataset
    config["train_split"] = os.path.join(root_path, "dataset/train/data")
    config["val_split"] = os.path.join(root_path, "dataset/val/data")
    config["test_split"] = os.path.join(root_path, "dataset/test_obs/data")

    # Data Loader setting
    config["workers"] = 24
    config["val_workers"] = config["workers"]

    # Training setting
    config["batch_size"] = 16
    config["val_batch_size"] = 16
    config["num_epochs"] = 100
    config["opt"] = "adam"
    config["lr"] = [1e-3, 1e-4]
    config["lr_epochs"] = [100]
    config["lr_func"] = StepLR(config["lr"], config["lr_epochs"])
    config["display_iters"] = 205942
    config["save_freq"] = 5
    config["display_iters"] = 1
    config["val_iters"] = 5
    config["cls_th"] = 2.0
    config["cls_ignore"] = 0.2
    config["mgn"] = 0.2
    config["cls_coef"] = 1.0
    config["reg_coef"] = 1.0
    config["training"] = True
    config["reactive"] = False

    # LaneGCN model setting
    config["n_actor"] = 128
    config["n_map"] = 128
    config["num_scales"] = 6
    config["num_mods"] = 6
    config["pred_size"] = 30
    config["pred_step"] = 1
    config["num_preds"] = config["pred_size"] // config["pred_step"]
    config["GAT_dropout"] = 0.5
    config["GAT_Leakyrelu_alpha"] = 0.2
    config["GAT_num_head"] = config["n_actor"]
    config["SRF_conv_num"] = 4


    file_path = os.path.abspath(os.curdir)
    model_name = os.path.basename(file_path).split(".")[0]
    config["save_dir"] = os.path.join(root_path, "SRFNet", "results", model_name)
    config["rot_aug"] = False

    return config