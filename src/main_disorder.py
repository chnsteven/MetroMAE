import argparse
from our_model import UcdGPT_model, MODEL_SIZE_CHOICES
from train import TrainLoop

import setproctitle
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from config.path_config import EXPERIMENT_PATH, LOG_PATH
from DataLoader import data_load_main_disorder
from train_utils import build_exp_tag, print_run_config
import torch as th
from torch.utils.tensorboard import SummaryWriter


def dev(device_id="0"):
    """
    Get the device to use for torch.distributed.
    #"""
    if th.cuda.is_available():
        return th.device("cuda:{}".format(device_id))
    return th.device("cpu")


MASK_STRATEGY_CHOICES = (
    "combined",
    "random_spatiotemporal",
    "cycle_aware",
    "spatio_gradient",
)


def create_argparser():
    defaults = dict(
        # experimental settings
        dataset="Crowd",
        disorder_dataset="event1",
        mode="training",  # training | testing
        process_name="process_name",
        his_len=24,
        pred_len=12,
        # model settings
        t_mask_ratio=1 / 16,
        s_mask_ratio=1 / 16,
        patch_size=1,
        t_patch_size=2,
        model_size="medium",
        no_qkv_bias=0,
        pos_emb="SinCos",
        # pretrain settings
        mask_strategy="combined",
        contrastive_weight=0.5,
        meta_weight=1.0,
        curriculum_mask=0,
        curriculum_mask_ratio=0.1,
        curriculum_mask_rate=2,
        fixed_mask_per_epoch=0,
        cycle_gamma=1.0,  # mask prob cap for cycle-aware BSF & spatio_gradient
        bsf_top_k=2,  # BehavioralStressFactor: dominant CWT periods per spatial location
        # training parameters
        lr=1e-3,
        min_lr=1e-5,
        early_stop=2,
        weight_decay=1e-6,
        log_interval=20,
        total_epoches=200,
        device_id="0",
        clip_grad=0.05,
        lr_anneal_steps=200,
        batch_size=128,
        hour_patch_size=1,
        eval_scope="full",  # full
        exp_root="", 
    )
    parser = argparse.ArgumentParser()
    for k, v in defaults.items():
        if k in ("mask_strategy", "model_size"):
            continue
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            from train_utils import str2bool

            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)

    parser.add_argument(
        "--model_size",
        default=defaults["model_size"],
        choices=MODEL_SIZE_CHOICES,
        help="Model size: medium (default) or large.",
    )
    parser.add_argument(
        "--mask_strategy",
        "--masking_strategy",
        dest="mask_strategy",
        default=defaults["mask_strategy"],
        choices=MASK_STRATEGY_CHOICES,
        help=(
            "Mask strategy: combined (random base + BSF|spatial meta), "
            "random_spatiotemporal, cycle_aware (BSF cycle mask), "
            "or spatio_gradient (spatial gradient only)"
        ),
    )
    return parser


th.multiprocessing.set_sharing_strategy("file_system")


def main():

    th.autograd.set_detect_anomaly(False)

    args = create_argparser().parse_args()
    setproctitle.setproctitle("{}-{}".format(args.process_name, args.device_id))

    data, test_data, val_data, args.scaler = data_load_main_disorder(args)
    args.dataset = args.disorder_dataset
    assert args.his_len + args.pred_len == args.seq_len

    exp_tag = build_exp_tag(args)
    if args.exp_root:
        exp_root = os.path.abspath(args.exp_root)
    else:
        exp_root = os.path.join(EXPERIMENT_PATH, exp_tag)
    os.makedirs(exp_root, exist_ok=True)

    event_name = args.disorder_dataset
    args.folder = os.path.join(exp_root, event_name)
    args.model_path = args.folder + os.sep
    logdir = os.path.join(LOG_PATH, exp_tag, event_name)

    os.makedirs(args.model_path, exist_ok=True)
    os.makedirs(args.model_path + "model_save/", exist_ok=True)

    writer = SummaryWriter(log_dir=logdir, flush_secs=5)
    device = dev(args.device_id)

    print_run_config(
        args,
        device=device,
        logdir=logdir,
    )

    model = UcdGPT_model(args=args).to(device)

    TrainLoop(
        args=args,
        writer=writer,
        model=model,
        data=data,
        test_data=test_data,
        val_data=val_data,
        device=device,
        early_stop=args.early_stop,
    ).run_loop()


if __name__ == "__main__":
    main()
