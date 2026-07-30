# -*- coding: utf-8 -*-

from __future__ import absolute_import

from ts_benchmark.report import report_csv


def report(report_config: dict, report_method: str = "csv") -> None:
    if report_method == "dash":
        try:
            from ts_benchmark.report import report_dash
        except ImportError as exc:
            raise ImportError(
                "report_dash is not available; use report_method='csv'"
            ) from exc
        report_dash.report(report_config)
    elif report_method == "csv":
        report_csv.report(report_config)
    else:
        raise ValueError(f"Unknown report method {report_method}")
