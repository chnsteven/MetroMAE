from ts_benchmark.baselines.MetroMAE._metromae_src import ensure_metromae_src

ensure_metromae_src()

from ts_benchmark.baselines.MetroMAE.adapter import MetroMAE

__all__ = ["MetroMAE"]
