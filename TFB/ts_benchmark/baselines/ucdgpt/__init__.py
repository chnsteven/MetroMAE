from ts_benchmark.baselines.ucdgpt._ucdgpt_src import ensure_ucdgpt_src

ensure_ucdgpt_src()

from ts_benchmark.baselines.ucdgpt.ucdgpt import UCDGPT

__all__ = ["UCDGPT"]
