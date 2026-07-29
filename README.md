# MetroMAE

MetroMAE is a masked autoencoder framework for spatiotemporal urban event forecasting. It combines behavioral stress factors, flexible masking strategies, and transformer-based reconstruction for multi-event city disorder prediction.

## Project Structure

```
MetroMAE/
├── config/          # Event labels and shared configuration
├── src/             # Model, training, evaluation, and figure scripts
│   ├── main_disorder.py
│   ├── our_model.py
│   ├── mask_strategy.py
│   ├── DataLoader.py
│   └── figure/
└── TFB/             # Time Series Benchmark integration and baselines
```

## Data

Place preprocessed event tensors under `Baselines/SH/eventx.npy` (or your configured dataset path). Event labels are defined in `config/sh_event_labels.json`.

Custom dataset loading is implemented in `src/DataLoader.py` via `data_load_myself(args)`.

## Installation

- Tested OS: Linux
- Python >= 3.9
- PyTorch >= 2.1.0
- TensorBoard

```bash
pip install -r requirements.txt
```

Install PyTorch with the CUDA build that matches your environment before installing the remaining dependencies.

## Training

From the repository root:

```bash
cd src
```

Example (combined mask strategy):

```bash
python main_disorder.py \
  --device_id 0 \
  --machine machine \
  --dataset event1 \
  --disorder_dataset event1 \
  --task short \
  --size middle \
  --mask_strategy combined \
  --lr 3e-4 \
  --prompt_ST 0 \
  --his_len 30 \
  --pred_len 2 \
  --t_patch_size 32 \
  --patch_size 8 \
  --total_epoches 100 \
  --t_mask_ratio 0.15 \
  --s_mask_ratio 0.15 \
  --contrastive_weight 0.5 \
  --log_interval 10 \
  --early_stop 5
```

Constraints:

- `T % t_patch_size == 0`
- `H % patch_size == 0`

## Model

- Core model: `src/our_model.py`
- Masking strategies: `src/mask_strategy.py`
- Loss configuration: `forward_loss(...)` in `src/our_model.py`

## Evaluation

```bash
cd src
python evaluate.py --help
```

## TFB Benchmark

Benchmark scripts and baselines live under `TFB/`. Experiment outputs are written to `TFB/results/` (gitignored).
