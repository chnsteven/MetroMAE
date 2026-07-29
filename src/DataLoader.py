import os
import random
import sys

import torch
import torch as th

_SRC_ROOT = os.path.dirname(os.path.abspath(__file__))
if _SRC_ROOT not in sys.path:
    sys.path.insert(0, _SRC_ROOT)
_BASELINES_ROOT = "/root/Baselines"
if _BASELINES_ROOT not in sys.path:
    sys.path.insert(0, _BASELINES_ROOT)

from common.sh_windows import prepare_sh_windows  # noqa: E402

AUTODL_TMP_ROOT = os.environ.get("AUTODL_TMP_ROOT", "/root/autodl-tmp")


def data_load(args):
    bundle = prepare_sh_windows(
        args.disorder_dataset,
        his_len=getattr(args, "his_len", None),
        pred_len=args.pred_len,
        hour_patch_size=getattr(args, "hour_patch_size", None),
    )

    args.seq_len = bundle.his_len + bundle.pred_len
    args.spatial_H = bundle.spatial_H
    args.spatial_W = bundle.spatial_W

    period_zero = torch.zeros_like(bundle.X_train[0])

    train_data = [
        [bundle.X_train[i], bundle.ts_train[i], period_zero]
        for i in range(len(bundle.X_train))
    ]
    val_data = [
        [bundle.X_val[i], bundle.ts_val[i], period_zero]
        for i in range(len(bundle.X_val))
    ]
    test_data = [
        [bundle.X_test[i], bundle.ts_test[i], period_zero]
        for i in range(len(bundle.X_test))
    ]

    batch_size = args.batch_size

    train_loader = th.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = th.utils.data.DataLoader(
        val_data,
        batch_size=4 * batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = th.utils.data.DataLoader(
        test_data,
        batch_size=4 * batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    my_scaler_disorder = bundle.scaler_event
    print(
        "min:",
        my_scaler_disorder._min,
        "max:",
        my_scaler_disorder._max,
        "(train-only fit)",
    )

    return train_loader, test_loader, val_loader, my_scaler_disorder


def data_load_disorder(args):

    data_all = []
    test_data_all = []
    val_data_all = []
    my_scaler_all = {}
    dataset_name = args.disorder_dataset
    data, test_data, val_data, my_scaler = data_load(args)
    data_all.append([dataset_name, data])
    test_data_all.append(test_data)
    val_data_all.append(val_data)
    my_scaler_all[dataset_name] = my_scaler

    data_all = [(name, i) for name, data in data_all for i in data]
    random.seed(1111)
    random.shuffle(data_all)

    return data_all, test_data_all, val_data_all, my_scaler_all


def data_load_main_disorder(args):

    data, test_data, val_data, scaler = data_load_disorder(args)

    return data, test_data, val_data, scaler
