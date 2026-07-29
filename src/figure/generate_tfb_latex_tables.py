#!/usr/bin/env python3
"""Generate LaTeX result tables directly from archived TFB workbooks.

Outputs:
  1. ``per_dataset.tex``: one row per dataset and forecasting horizon from
     ``overall_results.csv`` or archived TFB workbooks.
  2. ``per_horizon.tex``: horizon-wise means from ``per_horizon_results.csv``
     (see ``convert_per_horizon_to_latex.py``).
  3. ``table_ablation_results.tex``: horizon-wise MetroMAE ablations. Each cell
     is the mean of the eight event-level records saved in the corresponding
     archived TFB result file.

Usage:
    python src/figure/generate_tfb_latex_tables.py
    python src/figure/convert_per_horizon_to_latex.py
    python src/figure/convert_overall_results_to_latex.py
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
FIGURE_ROOT = REPO_ROOT / "src" / "figure"
if str(FIGURE_ROOT) not in sys.path:
    sys.path.insert(0, str(FIGURE_ROOT))

from _common import EVENT_LABELS  # noqa: E402

TFB_RESULTS = REPO_ROOT / "TFB" / "results"
OVERALL_RESULTS_FILE = TFB_RESULTS / "overall.csv"
PER_HORIZON_RESULTS_FILE = TFB_RESULTS / "per_horizon_results.csv"
OVERALL_RESULTS_CSV = TFB_RESULTS / "overall_results.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "AAAI27" / "Tables"
FIGURES_OUTPUT_DIR = REPO_ROOT / "AAAI27" / "Figures"
HORIZONS = ("d12", "d24", "d36", "d48")
HORIZON_LABELS = ("12-day", "24-day", "36-day", "48-day")
OVERALL_HORIZONS = (*HORIZONS, "ALL")
MODEL_SPECS = (
    ("AIR", "AIR", TFB_RESULTS / "AIR" / "air_result.xlsx"),
    ("GMAN", "GMAN", TFB_RESULTS / "GMAN" / "gman_result.xlsx"),
    ("PewLSTM", "PewLSTM", TFB_RESULTS / "PewLSTM" / "pewlstm_result.xlsx"),
    ("Prophet", "Prophet", TFB_RESULTS / "Prophet" / "prophet_result.xlsx"),
    ("UniST", "UniST", TFB_RESULTS / "UniST" / "unist_result.xlsx"),
    ("STMTM", "ST-MTM", TFB_RESULTS / "STMTM"),
    ("UCDGPT", "MetroMAE (ours)", TFB_RESULTS / "UCDGPT" / "ucdgpt_result.xlsx"),
)
ABLATION_SPECS = (
    ("UCDGPT", "MetroMAE (full)", TFB_RESULTS / "UCDGPT" / "ucdgpt_result.xlsx"),
    (
        "no_contra",
        "w/o contrastive loss",
        TFB_RESULTS / "UCDGPT-ablation-no_contra" / "hourly",
    ),
    (
        "no_random_mask",
        "w/o random base mask",
        TFB_RESULTS / "UCDGPT-ablation-no_random_mask" / "hourly",
    ),
    (
        "no_spatial_mask",
        "w/o spatial meta mask",
        TFB_RESULTS / "UCDGPT-ablation-no_spatial" / "hourly",
    ),
    (
        "no_temporal_mask",
        "w/o temporal meta mask",
        TFB_RESULTS / "UCDGPT-ablation-no_temporal" / "hourly",
    ),
)
EVENTS = tuple(EVENT_LABELS.keys())
CSV_DATASET_LABELS = tuple(EVENT_LABELS.values())
CSV_LABEL_TO_EVENT = {label: event for event, label in EVENT_LABELS.items()}
RESULT_CSV_HORIZONS = ("d12", "d24", "d36", "d48", "ALL")
RESULT_CSV_HORIZON_LABELS = ("12-day", "24-day", "36-day", "48-day", "All horizons")
METHOD_ALIASES = {
    "ST-MTMT": "STMTM",
    "ST-MTM": "STMTM",
    "UCDGPT(ours)": "UCDGPT",
    "UCDGPT (ours)": "UCDGPT",
    "MetroMAE(ours)": "UCDGPT",
    "MetroMAE (ours)": "UCDGPT",
}
HORIZON_LABEL_TO_KEY = {
    "12-day": "d12",
    "24-day": "d24",
    "36-day": "d36",
    "48-day": "d48",
    "All horizons": "ALL",
}


def event_name(file_name: object) -> str:
    """Normalize a TFB event filename such as ``event_3.csv`` to ``event3``."""
    match = re.search(r"event[_ -]?(\d+)", str(file_name).lower())
    if match is None:
        raise ValueError(f"Cannot determine dataset from file name: {file_name!r}")
    event = f"event{int(match.group(1))}"
    if event not in EVENTS:
        raise ValueError(f"Expected event0 through event7, got {event!r}")
    return event


def ranks(values: list[tuple[str, float]]) -> dict[str, int]:
    """Return lower-is-better ranks, preserving equal ranks for equal values."""
    ordered = sorted(values, key=lambda item: item[1])
    output: dict[str, int] = {}
    rank = 0
    for index, (model, value) in enumerate(ordered):
        if index and value != ordered[index - 1][1]:
            rank = index
        output[model] = rank
    return output


def format_value(value: float, rank: int) -> str:
    """Format lower-is-better ranks using bold/underline table emphasis."""
    text = f"{value:.4f}"
    if rank == 0:
        return f"\\textbf{{{text}}}"
    if rank == 1:
        return f"\\underline{{{text}}}"
    return text


def table_resize_width(full_width: bool) -> str:
    """Return LaTeX width for non-float tables in AAAI two-column layout."""
    return r"\textwidth" if full_width else r"\columnwidth"


def wrap_centered_table(
    caption: str,
    label: str,
    tabular: str,
    tabcolsep: str = "1.5pt",
    full_width: bool = False,
) -> str:
    """Wrap tabular content in a non-float center environment."""
    return "\n".join(
        [
            r"% Requires \usepackage{booktabs} and \usepackage{caption}.",
            r"\begin{center}",
            rf"\captionof{{table}}{{{caption}}}",
            rf"\label{{{label}}}",
            r"\scriptsize",
            rf"\setlength{{\tabcolsep}}{{{tabcolsep}}}",
            rf"\resizebox{{{table_resize_width(full_width)}}}{{!}}{{%",
            tabular + "%",
            r"}",
            r"\end{center}",
            "",
        ]
    )


def format_per_dataset_cells(
    data: pd.DataFrame,
    horizons: tuple[str, ...],
    model_names: list[str],
) -> dict[tuple[str, str, str, str], str]:
    """Format MAE/RMSE cells with lower-is-better ranks for each dataset--horizon--metric."""
    formatted: dict[tuple[str, str, str, str], str] = {}
    for horizon in horizons:
        for dataset in EVENTS:
            for metric in ("mae", "rmse"):
                rank_map = ranks(
                    [
                        (model, float(data.loc[(model, dataset, horizon), metric]))
                        for model in model_names
                    ]
                )
                for model in model_names:
                    formatted[(model, dataset, horizon, metric)] = format_value(
                        float(data.loc[(model, dataset, horizon), metric]),
                        rank_map[model],
                    )
    return formatted


def compute_model_win_counts(
    data: pd.DataFrame,
    horizons: tuple[str, ...],
    model_names: list[str],
) -> dict[str, dict[str, int]]:
    """Count Top-1 and Top-2 finishes per model across dataset--horizon--metric cells."""
    top1 = {model: 0 for model in model_names}
    top2 = {model: 0 for model in model_names}
    for horizon in horizons:
        for dataset in EVENTS:
            for metric in ("mae", "rmse"):
                rank_map = ranks(
                    [
                        (model, float(data.loc[(model, dataset, horizon), metric]))
                        for model in model_names
                    ]
                )
                for model in model_names:
                    rank = rank_map[model]
                    if rank == 0:
                        top1[model] += 1
                    elif rank == 1:
                        top2[model] += 1
    return {"top1": top1, "top2": top2}


def build_csv_layout_per_dataset_tabular(
    formatted: dict[tuple[str, str, str, str], str],
    horizons: tuple[str, ...],
    horizon_labels: tuple[str, ...],
    model_names: list[str],
    labels: dict[str, str],
    win_counts: dict[str, dict[str, int]] | None = None,
) -> str:
    """Build tabular matching ``overall_results.csv``: MAE horizons, then RMSE horizons."""
    n_horizons = len(horizons)
    n_metric_cols = 2 * n_horizons
    n_cols = 2 + n_metric_cols
    metric_col_spec = "c" * n_metric_cols

    rows: list[str] = []
    for model in model_names:
        for dataset_index, dataset in enumerate(EVENTS):
            method_cell = labels[model] if dataset_index == 0 else ""
            cells = [method_cell, EVENT_LABELS[dataset]]
            for horizon in horizons:
                cells.append(formatted[(model, dataset, horizon, "mae")])
            for horizon in horizons:
                cells.append(formatted[(model, dataset, horizon, "rmse")])
            rows.append(" & ".join(cells) + r" \\")

    footer: list[str] = []
    if win_counts is not None:
        top1_parts = [
            f"{labels[model]} ({win_counts['top1'][model]})" for model in model_names
        ]
        top2_parts = [
            f"{labels[model]} ({win_counts['top2'][model]})" for model in model_names
        ]
        footer = [
            r"\midrule",
            rf"\multicolumn{{{n_cols}}}{{l}}{{\textit{{Top-1 wins:}} {', '.join(top1_parts)}}} \\",
            rf"\multicolumn{{{n_cols}}}{{l}}{{\textit{{Top-2 wins:}} {', '.join(top2_parts)}}} \\",
        ]

    mae_start = 3
    mae_end = mae_start + n_horizons - 1
    rmse_start = mae_end + 1
    rmse_end = rmse_start + n_horizons - 1
    group_header = (
        " & & "
        + rf"\multicolumn{{{n_horizons}}}{{c}}{{MAE $\downarrow$}} & "
        + rf"\multicolumn{{{n_horizons}}}{{c}}{{RMSE $\downarrow$}}"
    )
    horizon_parts = ["", "", *horizon_labels, *horizon_labels]
    horizon_header = " & ".join(horizon_parts)

    return "\n".join(
        [
            rf"\begin{{tabular}}{{ll{metric_col_spec}}}",
            r"\toprule",
            group_header + r" \\",
            horizon_header + r" \\",
            rf"\cmidrule(lr){{{mae_start}-{mae_end}}} \cmidrule(lr){{{rmse_start}-{rmse_end}}}",
            r"\midrule",
            *rows,
            *footer,
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )


def wrap_per_dataset_table_output(
    tabular: str,
    caption: str,
    single_column: bool = True,
    full_width: bool = True,
    float_top: bool = False,
) -> str:
    if float_top and full_width:
        return "\n".join(
            [
                r"% Requires \usepackage{booktabs}.",
                r"\begin{table*}[t]",
                r"\centering",
                rf"\caption{{{caption}}}",
                r"\label{tab:per_dataset_results}",
                r"\scriptsize",
                r"\setlength{\tabcolsep}{1.5pt}",
                r"\resizebox{\textwidth}{!}{%",
                tabular + "%",
                r"}",
                r"\end{table*}",
                "",
            ]
        )

    if single_column:
        return wrap_centered_table(
            caption,
            "tab:per_dataset_results",
            tabular,
            tabcolsep="1.5pt",
            full_width=full_width,
        )

    return "\n".join(
        [
            r"% Requires \usepackage{booktabs}.",
            r"\begin{table*}[t]",
            r"\centering",
            rf"\caption{{{caption}}}",
            r"\label{tab:per_dataset_results}",
            r"\scriptsize",
            r"\setlength{\tabcolsep}{2pt}",
            tabular,
            r"\end{table*}",
            "",
        ]
    )


def load_horizon_frame(source: Path, horizon: str) -> pd.DataFrame:
    """Load one horizon sheet from an archived workbook or processed CSV directory."""
    if source.suffix == ".xlsx":
        return pd.read_excel(source, sheet_name=horizon)
    return pd.read_csv(source / f"{horizon}.csv")


def load_saved_overall_rows(model: str, source: Path) -> list[dict[str, object]]:
    """Load horizon-wise and ALL aggregates for models not present in overall.csv."""
    records: list[dict[str, object]] = []
    for horizon in HORIZONS:
        frame = load_horizon_frame(source, horizon)
        mean_rows = frame.loc[frame["file_name"].isna(), ["mae", "rmse"]]
        if mean_rows.empty:
            raise ValueError(f"{source}/{horizon} lacks a saved mean row")
        records.append(
            {
                "model_name": model,
                "horizon": horizon,
                "mae": float(mean_rows.iloc[0]["mae"]),
                "rmse": float(mean_rows.iloc[0]["rmse"]),
            }
        )
    records.append(
        {
            "model_name": model,
            "horizon": "ALL",
            "mae": float(pd.DataFrame(records)["mae"].mean()),
            "rmse": float(pd.DataFrame(records)["rmse"].mean()),
        }
    )
    return records


def load_per_dataset() -> pd.DataFrame:
    """Load raw event--horizon MAE/RMSE values without averaging them."""
    records: list[dict[str, object]] = []
    for model, _, source in MODEL_SPECS:
        if source.suffix == ".xlsx":
            workbook = pd.ExcelFile(source)
            for horizon in HORIZONS:
                if horizon not in workbook.sheet_names:
                    raise ValueError(f"{source} lacks sheet {horizon}")
        for horizon in HORIZONS:
            sheet = load_horizon_frame(source, horizon)
            required = {"file_name", "mae", "rmse"}
            if missing := required - set(sheet.columns):
                raise ValueError(f"{source}/{horizon} lacks {sorted(missing)}")
            for _, row in sheet.iterrows():
                if pd.isna(row["file_name"]):
                    continue
                records.append(
                    {
                        "model": model,
                        "dataset": event_name(row["file_name"]),
                        "horizon": horizon,
                        "mae": float(row["mae"]),
                        "rmse": float(row["rmse"]),
                    }
                )

    data = pd.DataFrame(records)
    expected = {
        (model, dataset, horizon)
        for model, _, _ in MODEL_SPECS
        for dataset in EVENTS
        for horizon in HORIZONS
    }
    found = set(
        data.loc[:, ["model", "dataset", "horizon"]].itertuples(index=False, name=None)
    )
    if missing_rows := expected - found:
        raise ValueError(f"Missing per-dataset TFB rows: {sorted(missing_rows)}")
    if data.duplicated(["model", "dataset", "horizon"]).any():
        raise ValueError("Duplicate model/dataset/horizon rows in TFB workbooks")
    return data.set_index(["model", "dataset", "horizon"]).sort_index()


def render_per_dataset_table(
    data: pd.DataFrame,
    single_column: bool = True,
    full_width: bool = True,
    float_top: bool = False,
) -> str:
    """Render raw per-event, per-horizon errors without aggregation."""
    model_names = [model for model, _, _ in MODEL_SPECS]
    labels = {model: label for model, label, _ in MODEL_SPECS}
    formatted = format_per_dataset_cells(data, HORIZONS, model_names)
    win_counts = compute_model_win_counts(data, HORIZONS, model_names)
    caption = (
        "Forecasting errors for every Urban Disorder event and prediction horizon. "
        "Values are reported without averaging across events or horizons. Each event "
        "is evaluated independently at 12, 24, 36, and 48 days; lower is better. "
        "Bold and underlined entries denote the best and second-best methods for each "
        "event--horizon--metric comparison, respectively. Top-1 and Top-2 win counts "
        "are summarized at the bottom of the table."
    )
    tabular = build_csv_layout_per_dataset_tabular(
        formatted,
        HORIZONS,
        HORIZON_LABELS,
        model_names,
        labels,
        win_counts=win_counts,
    )
    return wrap_per_dataset_table_output(
        tabular, caption, single_column, full_width, float_top=float_top
    )


def normalize_method_name(raw_name: str) -> str:
    """Map CSV method labels to internal model keys used by MODEL_SPECS."""
    name = raw_name.strip()
    return METHOD_ALIASES.get(name, name)


def load_overall_results_csv(path: Path) -> pd.DataFrame:
    """Load per-event results from ``overall_results.csv``.

    Layout (header row 1): Method | Dataset | 5 MAE horizons | 5 RMSE horizons.
    """
    frame = pd.read_csv(path, header=1)
    n_horizons = len(RESULT_CSV_HORIZONS)
    expected_cols = 2 + 2 * n_horizons
    if len(frame.columns) < expected_cols:
        raise ValueError(
            f"{path} must have Method, dataset, and {2 * n_horizons} metric columns; "
            f"got {list(frame.columns)}"
        )

    records: list[dict[str, object]] = []
    current_method: str | None = None
    for _, row in frame.iterrows():
        method_cell = str(row.iloc[0]).strip()
        if method_cell and method_cell.lower() != "nan":
            current_method = normalize_method_name(method_cell)

        dataset_label = str(row.iloc[1]).strip()
        if not dataset_label or dataset_label.lower() == "nan":
            continue
        if current_method is None:
            raise ValueError(f"{path} has dataset row before any method name")
        if dataset_label not in CSV_LABEL_TO_EVENT:
            raise ValueError(f"Unknown dataset label in {path}: {dataset_label!r}")

        dataset = CSV_LABEL_TO_EVENT[dataset_label]
        for horizon_index, horizon in enumerate(RESULT_CSV_HORIZONS):
            mae = float(row.iloc[2 + horizon_index])
            rmse = float(row.iloc[2 + n_horizons + horizon_index])
            records.append(
                {
                    "model": current_method,
                    "dataset": dataset,
                    "horizon": horizon,
                    "mae": mae,
                    "rmse": rmse,
                }
            )

    data = pd.DataFrame(records)
    models = [model for model, _, _ in MODEL_SPECS]
    expected = {
        (model, dataset, horizon)
        for model in models
        for dataset in EVENTS
        for horizon in RESULT_CSV_HORIZONS
    }
    found = set(
        data.loc[:, ["model", "dataset", "horizon"]].itertuples(index=False, name=None)
    )
    if missing_rows := expected - found:
        raise ValueError(f"Missing overall_results.csv rows: {sorted(missing_rows)}")
    if data.duplicated(["model", "dataset", "horizon"]).any():
        raise ValueError("Duplicate model/dataset/horizon rows in overall_results.csv")
    return data.set_index(["model", "dataset", "horizon"]).sort_index()


def render_per_event_results_table(
    data: pd.DataFrame,
    single_column: bool = True,
    full_width: bool = True,
    float_top: bool = True,
) -> str:
    """Render per-event errors matching ``overall_results.csv`` column layout."""
    model_names = [model for model, _, _ in MODEL_SPECS]
    labels = {model: label for model, label, _ in MODEL_SPECS}
    formatted = format_per_dataset_cells(data, RESULT_CSV_HORIZONS, model_names)
    win_counts = compute_model_win_counts(data, RESULT_CSV_HORIZONS, model_names)
    caption = (
        "Forecasting errors for every Urban Disorder event and prediction horizon. "
        "Values are reported without averaging across events or horizons. Each event "
        "is evaluated independently at 12, 24, 36, 48 days, and across all horizons; "
        "lower is better. Bold and underlined entries denote the best and second-best "
        "methods for each event--horizon--metric comparison, respectively. Top-1 and "
        "Top-2 win counts are summarized at the bottom of the table."
    )
    tabular = build_csv_layout_per_dataset_tabular(
        formatted,
        RESULT_CSV_HORIZONS,
        RESULT_CSV_HORIZON_LABELS,
        model_names,
        labels,
        win_counts=win_counts,
    )
    return wrap_per_dataset_table_output(
        tabular,
        caption,
        single_column=single_column,
        full_width=full_width,
        float_top=float_top,
    )


def load_per_horizon_csv(path: Path) -> pd.DataFrame:
    """Load horizon-wise means from ``per_horizon_results.csv``.

    Expected layout (row 0 may contain MAE/RMSE group headers):
      Method | 12-day ... All horizons | 12-day ... All horizons
    """
    frame = pd.read_csv(path, header=1)
    if len(frame.columns) != 11:
        raise ValueError(
            f"{path} must have 11 columns (Method + 5 MAE + 5 RMSE); got {list(frame.columns)}"
        )

    method_col = frame.columns[0]
    mae_columns = list(frame.columns[1:6])
    rmse_columns = list(frame.columns[6:11])
    expected_labels = ("12-day", "24-day", "36-day", "48-day", "All horizons")

    records: list[dict[str, object]] = []
    for _, row in frame.iterrows():
        model = normalize_method_name(str(row[method_col]))
        for label, mae_col, rmse_col in zip(expected_labels, mae_columns, rmse_columns):
            horizon = HORIZON_LABEL_TO_KEY[label]
            records.append(
                {
                    "model_name": model,
                    "horizon": horizon,
                    "mae": float(row[mae_col]),
                    "rmse": float(row[rmse_col]),
                }
            )

    data = pd.DataFrame(records)
    models = [model for model, _, _ in MODEL_SPECS]
    expected = {(model, horizon) for model in models for horizon in OVERALL_HORIZONS}
    found = set(
        data.loc[:, ["model_name", "horizon"]].itertuples(index=False, name=None)
    )
    if missing_rows := expected - found:
        raise ValueError(f"Missing per-horizon CSV rows: {sorted(missing_rows)}")
    return data.set_index(["model_name", "horizon"]).sort_index()


def load_overall() -> pd.DataFrame:
    """Read TFB's saved, already-aggregated horizon and ALL results verbatim."""
    path = OVERALL_RESULTS_FILE
    data = pd.read_excel(path)
    models = [model for model, _, _ in MODEL_SPECS]
    required = {"model_name", "horizon", "mae", "rmse"}
    if missing := required - set(data.columns):
        raise ValueError(f"{path} lacks {sorted(missing)}")
    data = data.loc[
        data["model_name"].isin(models) & data["horizon"].isin(OVERALL_HORIZONS)
    ].copy()
    expected = {(model, horizon) for model in models for horizon in OVERALL_HORIZONS}
    found = set(
        data.loc[:, ["model_name", "horizon"]].itertuples(index=False, name=None)
    )
    missing_models = {
        model for model in models if not any(model == row[0] for row in found)
    }
    if missing_rows := expected - found:
        supplemental: list[dict[str, object]] = []
        for model, _, source in MODEL_SPECS:
            if model not in missing_models:
                continue
            supplemental.extend(load_saved_overall_rows(model, source))
        if supplemental:
            data = pd.concat([data, pd.DataFrame(supplemental)], ignore_index=True)
            found = set(
                data.loc[:, ["model_name", "horizon"]].itertuples(
                    index=False, name=None
                )
            )
        if missing_rows := expected - found:
            raise ValueError(f"Missing saved TFB overall rows: {sorted(missing_rows)}")
    return data.set_index(["model_name", "horizon"]).sort_index()


def render_overall_table(data: pd.DataFrame, single_column: bool = False) -> str:
    models = [model for model, _, _ in MODEL_SPECS]
    labels = dict((model, label) for model, label, _ in MODEL_SPECS)
    metric_order = ("mae", "rmse")
    all_labels = (*HORIZON_LABELS, "All horizons")
    rows: list[str] = []
    for model in models:
        cells = [labels[model]]
        for metric in metric_order:
            for horizon in OVERALL_HORIZONS:
                rank_map = ranks(
                    [
                        (candidate, float(data.loc[(candidate, horizon), metric]))
                        for candidate in models
                    ]
                )
                cells.append(
                    format_value(
                        float(data.loc[(model, horizon), metric]), rank_map[model]
                    )
                )
        rows.append(" & ".join(cells) + r" \\")

    header = " & ".join(all_labels)
    caption = (
        "Horizon-wise mean forecasting errors across eight Urban Disorder "
        "event-category series. Lower values indicate better performance. "
        "Bold and underlined entries denote the best and second-best values "
        "in each column, respectively."
    )
    tabcolsep = "2pt" if single_column else "2.5pt"
    table_body = [
        r"\toprule",
        r"& \multicolumn{5}{c}{MAE} & \multicolumn{5}{c}{RMSE} \\",
        r"\cmidrule(lr){2-6} \cmidrule(lr){7-11}",
        "Method & " + header + " & " + header + r" \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
    ]
    tabular = "\n".join(
        [
            r"\begin{tabular}{lcccccccccc}",
            *table_body,
            r"\end{tabular}",
        ]
    )

    if single_column:
        return "\n".join(
            [
                r"% Requires \usepackage{booktabs} and \usepackage{caption}.",
                r"\begin{center}",
                rf"\captionof{{table}}{{{caption}}}",
                r"\label{tab:overall_results}",
                r"\scriptsize",
                rf"\setlength{{\tabcolsep}}{{{tabcolsep}}}",
                r"\resizebox{\columnwidth}{!}{%",
                tabular + "%",
                r"}",
                r"\end{center}",
                "",
            ]
        )

    return "\n".join(
        [
            # On a float-only page, remove the default stretch above a
            # double-column float so the requested top placement is literal.
            r"\makeatletter\setlength{\@dblfptop}{0pt}\makeatother",
            r"\begin{table*}[!t]",
            r"\centering",
            rf"\caption{{{caption}}}",
            r"\label{tab:overall_results}",
            r"\scriptsize",
            rf"\setlength{{\tabcolsep}}{{{tabcolsep}}}",
            tabular,
            r"\end{table*}",
            "",
        ]
    )


def load_ablation() -> pd.DataFrame:
    """Load the full model and four archived single-component ablations."""
    records: list[dict[str, object]] = []
    for variant, _, source in ABLATION_SPECS:
        for horizon in HORIZONS:
            if source.suffix == ".xlsx":
                frame = pd.read_excel(source, sheet_name=horizon)
            else:
                frame = pd.read_csv(source / f"{horizon}.csv")
            required = {"file_name", "mae", "rmse"}
            if missing := required - set(frame.columns):
                raise ValueError(f"{source}/{horizon} lacks {sorted(missing)}")
            frame = frame.loc[frame["file_name"].notna()].copy()
            events = [event_name(name) for name in frame["file_name"]]
            if sorted(events) != list(EVENTS):
                raise ValueError(
                    f"{source}/{horizon} must contain event0 through event7 exactly once"
                )
            records.append(
                {
                    "variant": variant,
                    "horizon": horizon,
                    "mae": float(frame["mae"].mean()),
                    "rmse": float(frame["rmse"].mean()),
                }
            )

    data = pd.DataFrame(records)
    expected = {
        (variant, horizon) for variant, _, _ in ABLATION_SPECS for horizon in HORIZONS
    }
    found = set(data.loc[:, ["variant", "horizon"]].itertuples(index=False, name=None))
    if missing_rows := expected - found:
        raise ValueError(f"Missing ablation rows: {sorted(missing_rows)}")
    return data.set_index(["variant", "horizon"]).sort_index()


def render_ablation_table(data: pd.DataFrame) -> str:
    """Render the compact, horizon-wise ablation table for the paper."""
    variants = [variant for variant, _, _ in ABLATION_SPECS]
    labels = {variant: label for variant, label, _ in ABLATION_SPECS}
    formatted: dict[tuple[str, str, str], str] = {}
    for horizon in HORIZONS:
        for metric in ("mae", "rmse"):
            rank_map = ranks(
                [
                    (variant, float(data.loc[(variant, horizon), metric]))
                    for variant in variants
                ]
            )
            for variant in variants:
                formatted[(variant, horizon, metric)] = format_value(
                    float(data.loc[(variant, horizon), metric]), rank_map[variant]
                )

    rows: list[str] = []
    for variant in variants:
        cells = [labels[variant]]
        for horizon in HORIZONS:
            cells.extend(
                (
                    formatted[(variant, horizon, "mae")],
                    formatted[(variant, horizon, "rmse")],
                )
            )
        rows.append(" & ".join(cells) + r" \\")

    horizon_header = " & ".join(
        rf"\multicolumn{{2}}{{c}}{{{label}}}" for label in HORIZON_LABELS
    )
    metric_header = " & ".join(
        ["MAE $\\downarrow$", "RMSE $\\downarrow$"] * len(HORIZONS)
    )
    return "\n".join(
        [
            r"% Requires \usepackage{booktabs}.",
            r"\begin{table*}[t]",
            r"\centering",
            r"\caption{Ablation results for MetroMAE. Each value is the mean MAE or RMSE across the eight Urban Disorder event series at the stated horizon. All variants use the same fixed-forecast protocol. Bold and underlined entries denote the best and second-best values within each metric--horizon column, respectively.}",
            r"\label{tab:ablation_results}",
            r"\small",
            r"\setlength{\tabcolsep}{4pt}",
            r"\begin{tabular}{lcccccccc}",
            r"\toprule",
            "Variant & " + horizon_header + r" \\",
            r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9}",
            "& " + metric_header + r" \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--per-horizon-csv",
        type=Path,
        default=None,
        help="Read horizon-wise means from per_horizon_results.csv instead of overall.csv",
    )
    parser.add_argument(
        "--only-overall",
        action="store_true",
        help="Generate only per_horizon.tex from per_horizon_results.csv",
    )
    parser.add_argument(
        "--no-pdf", action="store_true", help="Skip the per-dataset PDF preview"
    )
    return parser.parse_args()


def render_per_dataset_pdf(table_path: Path, output_path: Path) -> None:
    """Compile a landscape PDF preview of the per-dataset table."""
    if shutil.which("pdflatex") is None:
        raise RuntimeError("pdflatex is required to generate the per-dataset PDF")

    with tempfile.TemporaryDirectory(prefix="tfb_table_") as temp_dir:
        temp_path = Path(temp_dir)
        wrapper = temp_path / "per_dataset_table_preview.tex"
        wrapper.write_text(
            "\n".join(
                [
                    r"\documentclass[landscape]{article}",
                    r"\usepackage[margin=0.35in]{geometry}",
                    r"\usepackage{booktabs}",
                    r"\usepackage{xcolor}",
                    r"\begin{document}",
                    rf"\input{{{table_path}}}",
                    r"\end{document}",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        for _ in range(2):
            subprocess.run(
                [
                    "pdflatex",
                    "-interaction=nonstopmode",
                    "-halt-on-error",
                    wrapper.name,
                ],
                cwd=temp_path,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
            )
        shutil.copy2(temp_path / "per_dataset_table_preview.pdf", output_path)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    per_horizon_path = args.per_horizon_csv or (
        PER_HORIZON_RESULTS_FILE if args.only_overall else None
    )
    if per_horizon_path is not None:
        overall = render_overall_table(
            load_per_horizon_csv(per_horizon_path),
            single_column=True,
        )
        overall_path = args.output_dir / "per_horizon.tex"
        overall_path.write_text(overall, encoding="utf-8")
        print(f"Wrote {overall_path} from {per_horizon_path}")
        return

    per_dataset = render_per_dataset_table(load_per_dataset())
    overall = render_overall_table(load_overall())
    ablation = render_ablation_table(load_ablation())
    per_dataset_path = args.output_dir / "per_dataset.tex"
    overall_path = args.output_dir / "per_horizon.tex"
    ablation_path = args.output_dir / "table_ablation_results.tex"
    per_dataset_path.write_text(per_dataset, encoding="utf-8")
    overall_path.write_text(overall, encoding="utf-8")
    ablation_path.write_text(ablation, encoding="utf-8")
    print(f"Wrote {per_dataset_path}")
    print(f"Wrote {overall_path}")
    print(f"Wrote {ablation_path}")
    if not args.no_pdf:
        pdf_path = args.output_dir / "per_dataset.pdf"
        render_per_dataset_pdf(per_dataset_path.resolve(), pdf_path)
        print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
