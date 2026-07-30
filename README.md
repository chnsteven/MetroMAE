# MetroMAE

MetroMAE is a masked autoencoder framework for spatiotemporal urban event forecasting. It combines behavioral stress factors, flexible masking strategies, and transformer-based reconstruction for multi-event city disorder prediction.

## Project Structure

```
MetroMAE/
├── config/          # Path config and shared settings
├── src/             # Model, training, evaluation, and figure scripts
│   ├── main_disorder.py
│   ├── our_model.py
│   ├── mask_strategy.py
│   ├── DataLoader.py
│   ├── scripts/run.sh
│   └── figure/
└── TFB/             # Time Series Benchmark integration and baselines
```

## Data

Place preprocessed SH-Event as `event{0..7}.npy` under the configured data root

Event labels: `src/figure/event_label.json`.

Dataset loading: `src/DataLoader.py` → `preprocess.prepare_sh_windows`.

## Installation

- Tested OS: Linux
- Python >= 3.9
- PyTorch >= 2.1.0
- TensorBoard

```bash
pip install -r requirements.txt
```

Install PyTorch with the CUDA build that matches your environment before installing the remaining dependencies.

## Hyperparameter example

From `src/` (or via `src/scripts/run.sh`). Typical settings:

| Setting | Example | Notes |
|---|---|---|
| `mask_strategy` | `combined` | Also: `random_spatiotemporal`, `cycle_aware`, `spatio_gradient` |
| `his_len` / `pred_len` | `96` / `144` | In hour-patch steps; `seq_len = his_len + pred_len` |
| `hour_patch_size` | `6` | Must divide 24 |
| `t_patch_size` / `patch_size` | `16` / `4` | Require `T % t_patch_size == 0`, `H % patch_size == 0` |
| `model_size` | `medium` | Or `large` |
| `t_mask_ratio` / `s_mask_ratio` | `0.15` | Random spatiotemporal branch |
| `cycle_gamma` / `bsf_top_k` | `1.0` / `2` | Cycle-aware / spatial-gradient mask |
| `contrastive_weight` / `meta_weight` | `0.5` / `0.5` | Loss weights |
| `lr` / `min_lr` | `3e-4` / `1e-4` | AdamW + cosine anneal |

Constraints:

- `T % t_patch_size == 0`
- `H % patch_size == 0` (SH-Event grid is `8×8`)



```bash
cd src
bash scripts/run.sh
```



## TFB benchmark evaluation

Prepare TFB forecasting CSVs from SH-Event `.npy`, then link the dataset into TFB:

```bash
cd TFB
python scripts/convert_sh_to_tfb.py
python scripts/generate_forecast_meta.py
bash scripts/setup_forecasting_dataset_link.sh
```

Run MetroMAE under TFB (default smoke: event `0`, GPU `0`):

```bash
cd TFB
bash scripts/multivariate_forecast/SH_Event_script/MetroMAE.sh
```

Other baselines: `AIR.sh`, `GMAN.sh`, `PewLSTM.sh`, `Prophet.sh`, `STMTM.sh`, `UniST.sh` in the same script directory. Run all with `bash scripts/multivariate_forecast/SH_Event_script/run_all.sh`.

## Model

- Core model: `src/our_model.py`
- Masking strategies: `src/mask_strategy.py`
- Loss configuration: `forward_loss_patch_level(...)` in `src/our_model.py`
