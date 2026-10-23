import os
import random
import sys

# libgomp rejects 0. Fix it before torch loads OpenMP in this process or its workers.
_omp_threads = os.environ.get("OMP_NUM_THREADS", "")
if not _omp_threads.isdigit() or int(_omp_threads) < 1:
    os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import torch
import torch as th

_SRC_ROOT = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SRC_ROOT, ".."))
if _SRC_ROOT not in sys.path:
    sys.path.insert(0, _SRC_ROOT)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from preprocess import prepare_sh_windows  # noqa: E402
from priority_score import experiment_uses_cycle, precompute_cycle_cache  # noqa: E402


class _WindowDataset(th.utils.data.Dataset):
    def __init__(self, bundle, start: int, end: int):
        self.series = bundle.series
        self.starts = bundle.window_starts
        self.lo = int(start)
        self.hi = int(end)
        self.seq_len = bundle.seq_len
        self.steps_per_day = int(bundle.steps_per_day)

    def __len__(self):
        return self.hi - self.lo

    def __getitem__(self, j):
        index = self.lo + j
        s = int(self.starts[index])
        seq = self.seq_len
        window = np.ascontiguousarray(self.series[:, s : s + seq])
        x = torch.from_numpy(window)
        g = s + np.arange(seq, dtype=np.int64)
        ts = torch.empty((seq, 2), dtype=torch.long)
        ts[:, 0] = torch.from_numpy((g // self.steps_per_day) % 7)
        ts[:, 1] = torch.from_numpy((g % self.steps_per_day) * 2)
        return [x, ts, torch.zeros_like(x), torch.tensor(index, dtype=torch.long)]


def data_load(args):
    bundle = prepare_sh_windows(
        args.disorder_dataset,
        his_len=getattr(args, "his_len", None),
        pred_len=args.pred_len,
        hour_patch_size=getattr(args, "hour_patch_size", None),
    )

    args.hour_patch_size = bundle.hour_patch_size
    args.his_len = bundle.his_len
    args.pred_len = bundle.pred_len
    args.seq_len = bundle.his_len + bundle.pred_len
    args.spatial_H = bundle.spatial_H
    args.spatial_W = bundle.spatial_W

    train_data = _WindowDataset(bundle, 0, bundle.split_train_end)
    val_data = _WindowDataset(bundle, bundle.split_train_end, bundle.split_val_end)
    test_data = _WindowDataset(bundle, bundle.split_val_end, len(bundle.window_starts))
    sample = train_data[0]

    print(
        "[Data] hour_patch_size={}, his_len={}, pred_len={}, seq_len={}, sample_shape={}, n_train={}".format(
            args.hour_patch_size,
            args.his_len,
            args.pred_len,
            args.seq_len,
            tuple(sample[0].shape),
            len(train_data),
        )
    )

    batch_size = args.batch_size
    hour_patch = int(getattr(args, "hour_patch_size", 1) or 1)
    eval_batch_size = batch_size if hour_patch <= 1 else 4 * batch_size
    seed = int(getattr(args, "seed", 1111))
    generator = torch.Generator()
    generator.manual_seed(seed)

    def _init_worker(worker_id):
        worker_seed = (seed + worker_id) % (2**32)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    loader_kwargs = dict(
        num_workers=4,
        pin_memory=True,
        worker_init_fn=_init_worker,
    )
    train_loader = th.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        **loader_kwargs,
    )
    val_loader = th.utils.data.DataLoader(
        val_data,
        batch_size=eval_batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    test_loader = th.utils.data.DataLoader(
        test_data,
        batch_size=eval_batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    args.cycle_cache = None
    if experiment_uses_cycle(args):
        device = torch.device(
            "cuda:{}".format(args.device_id) if torch.cuda.is_available() else "cpu"
        )
        args.cycle_cache = precompute_cycle_cache(
            bundle.series,
            bundle.window_starts,
            bundle.his_len,
            getattr(args, "priority_k", 10),
            device,
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


class _NamedBatches:
    def __init__(self, name, loader):
        self.name = name
        self.loader = loader

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        for batch in self.loader:
            yield self.name, batch


def data_load_disorder(args):

    dataset_name = args.disorder_dataset
    data, test_data, val_data, my_scaler = data_load(args)
    data_all = _NamedBatches(dataset_name, data)
    test_data_all = [test_data]
    val_data_all = [val_data]
    my_scaler_all = {dataset_name: my_scaler}

    return data_all, test_data_all, val_data_all, my_scaler_all


def data_load_main_disorder(args):

    data, test_data, val_data, scaler = data_load_disorder(args)

    return data, test_data, val_data, scaler
