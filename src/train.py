import torch
from torch.optim import AdamW
import numpy as np
import torch.nn.functional as F
import math
import time

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text


def _panel(text: str) -> Panel:
    return Panel(text, title="UcdGPT", border_style="blue")


def _eval_panel_summary(
    rmse_key_result: dict,
    avg_rmse: float,
    best_rmse: float,
    early_stop: int,
    early_max: int,
    epoch: int,
) -> str:
    rows = [f"epoch {epoch + 1}"]
    for dataset_name, strategies in rmse_key_result.items():
        for _s, val in strategies.items():
            if isinstance(val, dict):
                rows.append(
                    f"  {dataset_name}: rmse={val['rmse']:.4f} mae={val['mae']:.4f}"
                )
    rows.append(
        f"  avg={avg_rmse:.4f}  best={best_rmse:.4f}  early={early_stop}/{early_max}"
    )
    return "\n".join(rows)


def _train_panel_summary(epoch, total_epochs, dataset_stats, batch_idx, n_batches):
    """Per-dataset training stats — same layout during batches and at epoch end."""
    rows = [f"epoch {epoch + 1}/{total_epochs}  batch {batch_idx}/{n_batches}"]
    for dataset_name in sorted(dataset_stats):
        stats = dataset_stats[dataset_name]
        if stats["num"] > 0:
            avg_loss = stats["loss"] / stats["num"]
            avg_rmse = np.sqrt(stats["sq_err"] / stats["num"])
            rows.append(f"  {dataset_name}: loss={avg_loss:.4f} rmse={avg_rmse:.4f}")
    return "\n".join(rows)


class LiveStatus:
    """Fixed bottom panel: training progress + last val/test summaries."""

    def __init__(self):
        self.console = Console()
        self._live = None
        self._train = "…"
        self._val = ""
        self._test = ""

    def __enter__(self):
        self._live = Live(_panel("…"), console=self.console, refresh_per_second=4)
        self._live.__enter__()
        self._render()
        return self

    def __exit__(self, *exc):
        if self._live is not None:
            self._live.__exit__(*exc)

    def _render(self) -> None:
        lines = ["[bold]Training[/]", self._train]
        if self._val:
            lines.extend(["", "[bold]Last VAL[/]", self._val])
        if self._test:
            lines.extend(["", "[bold]Last TEST[/]", self._test])
        if self._live is not None:
            self._live.update(_panel(Text.from_markup("\n".join(lines))))

    def set_train(self, text: str) -> None:
        self._train = text
        self._render()

    def set_eval(self, kind: str, text: str) -> None:
        if kind == "val":
            self._val = text
        else:
            self._test = text
        self._render()

    def gap(self, n: int = 2) -> None:
        """Blank lines above the panel (visual separation between log blocks)."""
        for _ in range(n):
            self.console.print()

    def detail(self, text: str) -> None:
        """Line(s) scrolling above the panel."""
        self.console.print(text)

    def detail_section(self, text: str, pad: int = 2) -> None:
        """Major eval block with extra vertical spacing."""
        self.gap(pad)
        self.console.print(text)
        self.gap(pad)


def _forecast_token_mask(mask: torch.Tensor, args) -> torch.Tensor:
    """Keep only masked positions in the prediction (future) time segment."""
    if getattr(args, "eval_scope", "full") != "forecast":
        return mask
    t_patch = args.t_patch_size
    his_t_patches = args.his_len // t_patch
    h_p = args.spatial_H // args.patch_size
    w_p = args.spatial_W // args.patch_size
    spatial_patches = h_p * w_p
    if mask.dim() == 2:
        l = mask.shape[1]
        idx = torch.arange(l, device=mask.device)
        forecast = (idx // spatial_patches) >= his_t_patches
        return mask * forecast.view(1, -1).to(mask.dtype)
    return mask


def _accumulate_mask_stats(agg: dict, mask_info: dict, weight: int) -> dict:
    if not mask_info or weight <= 0:
        return agg
    for branch, stats in mask_info.items():
        if not isinstance(stats, dict):
            continue
        bucket = agg.setdefault(
            branch,
            {
                "strategy": stats.get("strategy", branch),
                "t": 0.0,
                "s": 0.0,
                "u": 0.0,
                "n": 0,
            },
        )
        bucket["t"] += stats.get("t_mask_rate", 0.0) * weight
        bucket["s"] += stats.get("s_mask_rate", 0.0) * weight
        bucket["u"] += stats.get("union_rate", 0.0) * weight
        bucket["n"] += weight
    return agg


def _finalize_mask_stats(agg: dict) -> dict:
    out = {}
    for branch, v in agg.items():
        n = v["n"]
        if n <= 0:
            continue
        out[branch] = {
            "strategy": v["strategy"],
            "t_mask_rate": v["t"] / n,
            "s_mask_rate": v["s"] / n,
            "union_rate": v["u"] / n,
        }
    return out


def format_masking_line(mask_stats: dict) -> str:
    parts = []
    for branch, stats in mask_stats.items():
        strategy = stats.get("strategy", branch)
        if branch in ("meta", "base"):
            label = f"{branch}/{strategy}"
        else:
            label = strategy
        parts.append(
            f"{label}: t={stats['t_mask_rate']:.4f}, "
            f"s={stats['s_mask_rate']:.4f}, union={stats['union_rate']:.4f}"
        )
    return "[Masking] " + " | ".join(parts)


def evaluate_loader(
    model,
    args,
    data_loader,
    dataset_name,
    device,
    *,
    mask_strategy=None,
    seed=None,
    eval_type="test",
    collect_arrays=False,
):
    """
    Global RMSE/MAE on inverse-scaled masked positions (event-only branch).

    Matches TrainLoop.Sample pooling: sum errors globally, divide by N.

    When ``eval_mask_strategy`` is set on ``args``, it overrides
    ``args.mask_strategy`` unless ``mask_strategy`` is passed explicitly.
    """
    if mask_strategy is None:
        mask_strategy = getattr(args, "eval_mask_strategy", None) or args.mask_strategy

    model.eval()
    sq_error_sum = 0.0
    abs_error_sum = 0.0
    loss_norm_sum = 0.0
    num = 0
    pred_chunks = []
    true_chunks = []
    mask_agg = {}

    with torch.no_grad():
        for batch in data_loader:
            batch = [item.to(device, non_blocking=True) for item in batch]
            loss, loss2, pred, target, mask = model(
                batch,
                mask_strategy=mask_strategy,
                seed=seed,
                data=dataset_name,
                mode="forward",
            )

            if isinstance(loss2, dict) and loss2.get("mask_info"):
                _accumulate_mask_stats(mask_agg, loss2["mask_info"], mask.numel())

            mask = _forecast_token_mask(mask, args)

            pred = torch.clamp(pred, min=-1, max=1)
            pred_mask = pred[mask == 1]
            target_mask = target[mask == 1]

            pred_real = args.scaler[dataset_name].inverse_transform(
                pred_mask.reshape(-1, 1).detach().cpu().numpy()
            )
            target_real = args.scaler[dataset_name].inverse_transform(
                target_mask.reshape(-1, 1).detach().cpu().numpy()
            )

            elem_count = pred_real.size
            sq_error_sum += np.sum((pred_real - target_real) ** 2)
            abs_error_sum += np.sum(np.abs(pred_real - target_real))
            loss_norm_sum += loss.item() * elem_count
            num += elem_count

            if collect_arrays:
                pred_chunks.append(pred_real.reshape(-1))
                true_chunks.append(target_real.reshape(-1))

    mask_stats = _finalize_mask_stats(mask_agg)

    if num == 0:
        rmse = 0.0
        mae = 0.0
        loss_test = 0.0
    else:
        mse = sq_error_sum / num
        rmse = float(np.sqrt(mse))
        mae = float(abs_error_sum / num)
        loss_test = float(loss_norm_sum / num)

    if collect_arrays:
        pred_all = (
            np.concatenate(pred_chunks)
            if pred_chunks
            else np.array([], dtype=np.float64)
        )
        true_all = (
            np.concatenate(true_chunks)
            if true_chunks
            else np.array([], dtype=np.float64)
        )
        return rmse, mae, loss_test, num, pred_all, true_all, mask_stats

    return rmse, mae, loss_test, num, mask_stats


class TrainLoop:
    def __init__(
        self,
        args,
        writer,
        model,
        data,
        test_data,
        val_data,
        device,
        early_stop=5,
    ):
        self.args = args
        self.writer = writer
        self.model = model
        self.data = data
        self.test_data = test_data
        self.val_data = val_data
        self.device = device
        self.stop_training = False
        self.lr_anneal_steps = args.lr_anneal_steps
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.opt = AdamW(
            [p for p in self.model.parameters() if p.requires_grad == True],
            lr=args.lr,
            weight_decay=self.weight_decay,
        )
        self.log_interval = args.log_interval
        self.warmup_steps = 10
        self.min_lr = args.min_lr
        self.best_val_rmse = 1e9  # best val RMSE
        self.best_test_rmse = 1e9  # best test RMSE
        self.early_stop_counter = 0  # epochs w/o improvement
        self.curriculum_improve_counter = 0  # val best refresh count
        self.last_lr = args.lr

    def _epoch_train_log_path(self, dataset_name):
        return self.args.model_path + f"epoch_train_{dataset_name}.log"

    def _eval_log_path(self, dataset_name):
        return self.args.model_path + f"eval_{dataset_name}.log"

    def _append_epoch_train_log(self, dataset_name, line):
        with open(self._epoch_train_log_path(dataset_name), "a", encoding="utf-8") as f:
            f.write(line if line.endswith("\n") else line + "\n")

    def _append_eval_log(self, dataset_name, line):
        with open(self._eval_log_path(dataset_name), "a", encoding="utf-8") as f:
            f.write(line if line.endswith("\n") else line + "\n")

    def _format_eval_result(
        self, dataset_name, dataset_result, tag, step, for_log=False
    ):
        """Format per-dataset eval metrics as a single readable line."""
        parts = []
        for strategy, val in dataset_result.items():
            if isinstance(val, dict) and "rmse" in val:
                r = val.get("rmse", float("inf"))
                m = val.get("mae", float("inf"))
                parts.append(f"{strategy}: rmse={r:.6f}, mae={m:.6f}")
        detail = " | ".join(parts) if parts else str(dataset_result)
        line = f"[{dataset_name}] {tag} epoch:{step} | {detail}"
        if for_log:
            train_time = getattr(self, "last_epoch_training_time", "N/A")
            line += f" | train_time:{train_time}min"
        return line

    def run_step(self, batch, step, mask_strategy, index, name, mask_seed=None):
        self.opt.zero_grad()
        loss, num, sq_err, num2, loss_components = self.forward_backward(
            batch,
            step,
            mask_strategy,
            index=index,
            name=name,
            mask_seed=mask_seed,
        )

        self._anneal_lr()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=self.args.clip_grad
        )
        self.opt.step()
        return loss, num, sq_err, num2, loss_components, grad_norm.item()

    def Sample(
        self,
        test_data,
        step,
        mask_strategy,
        seed=None,
        dataset="",
        index=0,
        Type="val",
    ):
        rmse, mae, loss_test, _, mask_stats = evaluate_loader(
            self.model,
            self.args,
            test_data[index],
            dataset,
            self.device,
            mask_strategy=mask_strategy,
            seed=seed,
            eval_type=Type,
        )
        return rmse, mae, loss_test, mask_stats

    def Evaluation(self, test_data, epoch, seed=None, best=True, Type="val", ui=None):
        lines = [
            f"{Type.upper()} Evaluation - Epoch {epoch + 1}",
            "=" * 60,
        ]

        rmse_list = []
        rmse_key_result = {}

        # Dataset names from scaler dict (supports used_data='multi')
        if hasattr(self.args, "scaler") and isinstance(self.args.scaler, dict):
            dataset_names = list(self.args.scaler.keys())
        else:
            # Fallback: split args.dataset
            dataset_names = self.args.dataset.split("*")

        for index, dataset_name in enumerate(dataset_names):
            rmse_key_result[dataset_name] = {}

            s = self.args.mask_strategy
            rmse, mae, loss_test, mask_stats = self.Sample(
                test_data,
                epoch,
                mask_strategy=s,
                seed=seed,
                dataset=dataset_name,
                index=index,
                Type=Type,
            )
            rmse_list.append(rmse)
            rmse_key_result[dataset_name][s] = {"rmse": rmse, "mae": mae}
            lines.append(f"  [{dataset_name}] {s}: RMSE={rmse:.6f}, MAE={mae:.6f}")
            if mask_stats:
                masking_line = format_masking_line(mask_stats)
                lines.append(f"  {masking_line}")
                self._append_eval_log(dataset_name, masking_line)

            if Type == "val":
                self.writer.add_scalar(
                    "Evaluation/{}-{}".format(dataset_name.split("_C")[0], s),
                    rmse,
                    epoch,
                )
            elif Type == "test":
                self.writer.add_scalar(
                    "Test_RMSE/{}-{}".format(dataset_name.split("_C")[0], s),
                    rmse,
                    epoch,
                )
                self.writer.add_scalar(
                    "Test_MAE/{}-{}".format(dataset_name.split("_C")[0], s),
                    mae,
                    epoch,
                )

        avg_rmse = np.mean(rmse_list)
        lines.append("=" * 60)
        full = "\n".join(lines)

        if ui is not None:
            ui.detail_section(full)
        else:
            print(full)

        if best:
            self.best_model_save(epoch, avg_rmse, rmse_key_result, Type=Type, ui=ui)

        if ui is not None:
            best_rmse = self.best_val_rmse if Type == "val" else self.best_test_rmse
            ui.set_eval(
                Type,
                _eval_panel_summary(
                    rmse_key_result,
                    avg_rmse,
                    best_rmse,
                    self.early_stop_counter,
                    self.args.early_stop,
                    epoch,
                ),
            )
        return avg_rmse, rmse_key_result

    def best_model_save(self, step, rmse, rmse_key_result, Type="val", ui=None):
        """
        Save best model; track best val or test RMSE by Type.

        Args:
            step: epoch idx
            rmse: true RMSE (denormed)
            rmse_key_result: RMSE result dict
            Type: "val" or "test"
        """
        if Type == "val":
            best_rmse = self.best_val_rmse
            scalar_name = "Evaluation/Val_RMSE_best"
        elif Type == "test":
            best_rmse = self.best_test_rmse
            scalar_name = "Evaluation/Test_RMSE_best"
        else:
            best_rmse = self.best_val_rmse
            scalar_name = "Evaluation/RMSE_best"

        if rmse < best_rmse:
            self.early_stop_counter = 0
            torch.save(
                self.model.state_dict(),
                self.args.model_path + "model_save/model_best.pkl",
            )

            if Type == "val":
                self.best_val_rmse = rmse
            elif Type == "test":
                self.best_test_rmse = rmse

            self.writer.add_scalar(scalar_name, rmse, step)
            if ui is not None:
                ui.gap(1)
                ui.detail(f"{Type.upper()}_RMSE_best: {rmse}")

            for dataset_name, dataset_result in rmse_key_result.items():
                log_line = self._format_eval_result(
                    dataset_name,
                    dataset_result,
                    f"{Type.upper()}_best",
                    step,
                    for_log=True,
                )
                display_line = self._format_eval_result(
                    dataset_name, dataset_result, f"{Type.upper()}_best", step
                )
                result_file = self.args.model_path + f"result_{dataset_name}.txt"
                with open(result_file, "w", encoding="utf-8") as f:
                    f.write(log_line + "\n")
                if ui is not None:
                    ui.detail(display_line)
                else:
                    print(display_line)
                self._append_eval_log(dataset_name, log_line)

            return "save"

        else:
            self.early_stop_counter += 1
            if Type == "val":
                status = (
                    f"Val_RMSE: {rmse}, Val_RMSE_best: {self.best_val_rmse}, "
                    f"early_stop: {self.early_stop_counter}/{self.args.early_stop}"
                )
            elif Type == "test":
                status = (
                    f"Test_RMSE: {rmse}, Test_RMSE_best: {self.best_test_rmse}, "
                    f"early_stop: {self.early_stop_counter}/{self.args.early_stop}"
                )
            else:
                raise ValueError(f"Invalid Type: {Type}")
            if ui is not None:
                ui.gap(1)
                ui.detail(status)
            else:
                print(status)

            for dataset_name in rmse_key_result.keys():
                self._append_eval_log(
                    dataset_name,
                    f"[{dataset_name}] {Type.upper()}_not_improved rmse:{rmse:.6f}, "
                    f"early_stop:{self.early_stop_counter}/{self.args.early_stop}",
                )

            if self.early_stop_counter >= self.args.early_stop:
                if ui is not None:
                    ui.detail("Early stop!")
                else:
                    print("Early stop!")
                for dataset_name in rmse_key_result.keys():
                    self._append_eval_log(dataset_name, f"[{dataset_name}] Early stop!")
                self.stop_training = True
                return

    def mask_select(self):
        return self.args.mask_strategy

    def _maybe_bump_curriculum_mask(self, epoch, val_rmse, prev_best_val_rmse):
        if not getattr(self.args, "curriculum_mask", 0):
            return
        if val_rmse >= prev_best_val_rmse:
            return

        rate = self.args.curriculum_mask_rate
        if rate < 1:
            return

        self.curriculum_improve_counter += 1
        if self.curriculum_improve_counter % rate != 0:
            return

        cap = 0.75
        step = self.args.curriculum_mask_ratio
        new_t = min(cap, self.args.t_mask_ratio + step)
        new_s = min(cap, self.args.s_mask_ratio + step)
        if new_t <= self.args.t_mask_ratio and new_s <= self.args.s_mask_ratio:
            return

        self.args.t_mask_ratio = new_t
        self.args.s_mask_ratio = new_s
        self.writer.add_scalar("Curriculum/t_mask_ratio", new_t, epoch)
        self.writer.add_scalar("Curriculum/s_mask_ratio", new_s, epoch)

    def run_loop(self):
        step = 0

        if self.args.mode == "testing":
            self.Evaluation(self.val_data, 0, best=True, Type="val")
            exit()

        with LiveStatus() as ui:
            self.Evaluation(self.val_data, 0, best=True, Type="val", ui=ui)

            for epoch in range(self.args.total_epoches):
                if self.stop_training:
                    break

                self.step = epoch

                dataset_stats = {}
                loss_component_sums = {
                    "loss_base": 0.0,
                    "loss_meta": 0.0,
                    "loss_contra": 0.0,
                }
                loss_component_batches = 0
                grad_norm_sum = 0.0
                grad_norm_batches = 0
                loss_all, num_all, sq_err_all, num_all2 = 0.0, 0.0, 0.0, 0.0
                start = time.time()
                n_batches = len(self.data)
                fixed_mask = bool(getattr(self.args, "fixed_mask_per_epoch", 0))

                for batch_idx, (name, batch) in enumerate(self.data):
                    if name not in dataset_stats:
                        dataset_stats[name] = {
                            "loss": 0.0,
                            "num": 0.0,
                            "sq_err": 0.0,
                            "time": 0.0,
                            "batches": 0,
                        }

                    # Per-dataset timing
                    t0 = time.time()

                    mask_strategy = self.mask_select()
                    mask_seed = (epoch * 100000 + batch_idx) if fixed_mask else None
                    loss, num, sq_err, num2, loss_components, grad_norm = self.run_step(
                        batch,
                        step,
                        mask_strategy=mask_strategy,
                        index=0,
                        name=name,
                        mask_seed=mask_seed,
                    )
                    grad_norm_sum += grad_norm
                    grad_norm_batches += 1

                    if loss_components:
                        for key in loss_component_sums:
                            loss_component_sums[key] += loss_components.get(key, 0.0)
                        loss_component_batches += 1

                    t1 = time.time()
                    dataset_stats[name]["time"] += t1 - t0  # accumulate per-batch time
                    dataset_stats[name]["batches"] += 1

                    step += 1

                    # dataset-specific statistics
                    dataset_stats[name]["loss"] += loss * num
                    dataset_stats[name]["num"] += num
                    dataset_stats[name]["sq_err"] += sq_err

                    # global stats
                    loss_all += loss * num
                    sq_err_all += sq_err
                    num_all += num
                    num_all2 += num2

                    ui.set_train(
                        _train_panel_summary(
                            epoch,
                            self.args.total_epoches,
                            dataset_stats,
                            batch_idx + 1,
                            n_batches,
                        )
                    )

                end = time.time()
                total_training_time = round((end - start) / 60.0, 2)
                self.last_epoch_training_time = total_training_time

                if num_all > 0:
                    self.writer.add_scalar(
                        "Training/Loss_epoch",
                        np.sqrt(sq_err_all / num_all),
                        epoch,
                    )
                if grad_norm_batches > 0:
                    self.writer.add_scalar(
                        "Training/grad_norm_epoch",
                        grad_norm_sum / grad_norm_batches,
                        epoch,
                    )
                self.writer.add_scalar("Training/LR", self.last_lr, epoch)
                if loss_component_batches > 0:
                    for key, total in loss_component_sums.items():
                        self.writer.add_scalar(
                            f"Training/{key}_epoch",
                            total / loss_component_batches,
                            epoch,
                        )

                for dataset_name, stats in dataset_stats.items():
                    if stats["num"] > 0:
                        dataset_training_time = round(stats["time"] / 60.0, 2)
                        avg_loss = stats["loss"] / stats["num"]
                        avg_rmse = np.sqrt(stats["sq_err"] / stats["num"])
                        log_line = (
                            f"[{dataset_name}] epoch:{epoch + 1}, batches:{stats['batches']}, "
                            f"training loss:{avg_loss:.6f}, training rmse:{avg_rmse:.6f}, "
                            f"time:{dataset_training_time}min"
                        )
                        self._append_epoch_train_log(dataset_name, log_line)

                ui.set_train(
                    _train_panel_summary(
                        epoch,
                        self.args.total_epoches,
                        dataset_stats,
                        n_batches,
                        n_batches,
                    )
                )

                if (
                    epoch % self.log_interval == 0
                    and epoch > 0
                    or epoch == 10
                    or epoch == self.args.total_epoches - 1
                ):
                    ui.gap(3)
                    prev_best_val_rmse = self.best_val_rmse
                    rmse_val, rmse_key_val = self.Evaluation(
                        self.val_data, epoch, best=True, Type="val", ui=ui
                    )

                    saved = self.best_val_rmse < prev_best_val_rmse

                    self._maybe_bump_curriculum_mask(
                        epoch, rmse_val, prev_best_val_rmse
                    )

                    if saved:
                        ui.gap(2)
                        self.Evaluation(
                            self.test_data, epoch, best=True, Type="test", ui=ui
                        )
                    ui.gap(3)

                if self.stop_training:
                    ui.detail("Training terminated after early stopping.")
                    break

    def model_forward(
        self,
        batch,
        model,
        mask_strategy,
        seed=None,
        data=None,
        mode="backward",
    ):
        batch = [i.to(self.device, non_blocking=True) for i in batch]

        loss, loss2, pred, target, mask = self.model(
            batch,
            mask_strategy=mask_strategy,
            seed=seed,
            data=data,
            mode=mode,
        )
        return loss, loss2, pred, target, mask

    def forward_backward(
        self,
        batch,
        step,
        mask_strategy,
        index,
        name=None,
        mask_seed=None,
    ):
        loss, loss_components, pred, target, mask = self.model_forward(
            batch,
            self.model,
            mask_strategy,
            seed=mask_seed,
            data=name,
            mode="backward",
        )

        # Match eval: clamp to norm range then denorm
        pred = torch.clamp(pred, min=-1, max=1)

        # pred/target are (N, L, patch_num); no squeeze needed
        pred_mask = pred[mask == 1]
        target_mask = target[mask == 1]

        # inverse_transform on GPU; avoid GPU->CPU->GPU
        # inverse: X = (x + 1) / 2 * (max - min) + min
        scaler = self.args.scaler[name]
        _min = torch.tensor(scaler._min, device=pred_mask.device, dtype=pred_mask.dtype)
        _max = torch.tensor(scaler._max, device=pred_mask.device, dtype=pred_mask.dtype)
        pred_real = (pred_mask + 1.0) / 2.0 * (_max - _min) + _min
        target_real = (target_mask + 1.0) / 2.0 * (_max - _min) + _min

        # Batch MSE/RMSE; accumulate sq err like eval
        mse_mean = F.mse_loss(pred_real, target_real, reduction="mean")
        rmse = torch.sqrt(mse_mean)
        count = pred_real.numel()
        sq_error_sum = mse_mean.item() * count

        loss.backward()

        logged_components = {}
        if isinstance(loss_components, dict):
            for key, value in loss_components.items():
                if key == "mask_info" or not hasattr(value, "detach"):
                    continue
                logged_components[key] = value.detach().item()
        # Count unmasked elems; avoid bitwise NOT on non-bool tensor
        unmasked_count = (mask == 0).sum().item()
        return loss.item(), count, sq_error_sum, unmasked_count, logged_components

    def _anneal_lr(self):
        if self.step < self.warmup_steps:
            lr = self.lr * (self.step + 1) / self.warmup_steps
        elif self.step < self.lr_anneal_steps:
            lr = self.min_lr + (self.lr - self.min_lr) * 0.5 * (
                1.0
                + math.cos(
                    math.pi
                    * (self.step - self.warmup_steps)
                    / (self.lr_anneal_steps - self.warmup_steps)
                )
            )
        else:
            lr = self.min_lr
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr
        self.last_lr = lr
        return lr
