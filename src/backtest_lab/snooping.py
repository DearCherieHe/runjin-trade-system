from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def data_snooping_audit_frame(spec: dict[str, Any], stats: dict[str, Any], bars: int) -> pd.DataFrame:
    config = spec.get("snooping_check", {}) or {}
    max_parameters = int(config.get("max_parameters", 5))
    min_bars_per_parameter = int(config.get("min_bars_per_parameter", 30))
    assumed_trials = max(1, int(config.get("assumed_trials", 1)))
    parameter_count, parameter_notes = count_adjustable_parameters(spec)
    qualitative_count = count_qualitative_choices(spec)
    bars_per_parameter = bars / max(parameter_count, 1)
    sharpe = _coerce_float(stats.get("Sharpe Ratio"))
    deflated_sharpe_proxy = None
    if sharpe is not None:
        deflated_sharpe_proxy = sharpe / np.sqrt(1 + np.log(assumed_trials))
    ge_zero_status, ge_zero_threshold, ge_zero_reason = _sample_size_true_sharpe_ge_zero(sharpe, bars)
    ge_one_status, ge_one_threshold, ge_one_reason = _sample_size_true_sharpe_ge_one(sharpe, bars)

    rows = [
        {
            "check": "adjustable_parameter_count",
            "status": _status_parameter_count(parameter_count, max_parameters),
            "value": parameter_count,
            "threshold": f"<= {max_parameters}",
            "reason": "; ".join(parameter_notes) or "No adjustable parameters detected.",
        },
        {
            "check": "bars_per_parameter",
            "status": "pass" if bars_per_parameter >= min_bars_per_parameter else "review",
            "value": round(bars_per_parameter, 1),
            "threshold": f">= {min_bars_per_parameter}",
            "reason": "More independent history per parameter reduces overfit risk.",
        },
        {
            "check": "assumed_trials",
            "status": _status_trials(assumed_trials),
            "value": assumed_trials,
            "threshold": "<= 5 preferred",
            "reason": "Repeated tweaks, scans, and qualitative choices should be counted as trials.",
        },
        {
            "check": "qualitative_choice_count",
            "status": "pass" if qualitative_count <= 3 else "review",
            "value": qualitative_count,
            "threshold": "<= 3 preferred",
            "reason": "Template, execution timing, cost model, slippage model, and position model choices can also be optimized implicitly.",
        },
        {
            "check": "deflated_sharpe_proxy",
            "status": _status_deflated_sharpe(deflated_sharpe_proxy),
            "value": round(deflated_sharpe_proxy, 3) if deflated_sharpe_proxy is not None else "N/A",
            "threshold": ">= 0.5 preferred",
            "reason": "Conservative Sharpe haircut using assumed_trials. This is an explainable proxy, not the exact Bailey formula.",
        },
        {
            "check": "sample_size_true_sharpe_ge_0",
            "status": ge_zero_status,
            "value": f"bars={bars}, sharpe={round(sharpe, 3) if sharpe is not None else 'N/A'}",
            "threshold": ge_zero_threshold,
            "reason": ge_zero_reason,
        },
        {
            "check": "sample_size_true_sharpe_ge_1",
            "status": ge_one_status,
            "value": f"bars={bars}, sharpe={round(sharpe, 3) if sharpe is not None else 'N/A'}",
            "threshold": ge_one_threshold,
            "reason": ge_one_reason,
        },
    ]
    return pd.DataFrame(rows)


def count_adjustable_parameters(spec: dict[str, Any]) -> tuple[int, list[str]]:
    params = spec.get("parameters", {}) or {}
    notes = []
    count = 0
    if isinstance(params, dict):
        count += len(params)
        if params:
            notes.append(f"strategy parameters={len(params)}")
    for field in ["position_size", "stop_loss_pct", "take_profit_pct"]:
        value = spec.get(field)
        if value is not None and _coerce_float(value) not in {None, 0.0}:
            count += 1
            notes.append(field)
    position_params = spec.get("position_parameters", {}) or {}
    if isinstance(position_params, dict):
        count += len(position_params)
        if position_params:
            notes.append(f"position parameters={len(position_params)}")
    return count, notes


def count_qualitative_choices(spec: dict[str, Any]) -> int:
    fields = ["template", "trade_on_close", "commission_model", "slippage_model", "position_model", "benchmark"]
    return sum(1 for field in fields if field in spec and str(spec.get(field, "")).strip() != "")


def _status_parameter_count(parameter_count: int, max_parameters: int) -> str:
    if parameter_count <= max_parameters:
        return "pass"
    if parameter_count <= max_parameters + 2:
        return "review"
    return "fail"


def _status_trials(assumed_trials: int) -> str:
    if assumed_trials <= 5:
        return "pass"
    if assumed_trials <= 25:
        return "review"
    return "fail"


def _status_deflated_sharpe(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "review"
    if value >= 0.5:
        return "pass"
    if value >= 0:
        return "review"
    return "fail"


def _sample_size_true_sharpe_ge_zero(sharpe: float | None, bars: int) -> tuple[str, str, str]:
    if sharpe is None or not np.isfinite(sharpe):
        return "review", "Sharpe >= 1 with 681 bars, or Sharpe >= 2 with 174 bars", "Sharpe is unavailable, so minimum sample confidence cannot be evaluated."
    if sharpe >= 2:
        threshold = 174
        if bars >= threshold:
            return "pass", "Sharpe >= 2 and bars >= 174", "Meets the practical threshold for confidence that true Sharpe is at least 0."
        return "review", "Sharpe >= 2 and bars >= 174", "High backtest Sharpe helps, but the sample is still shorter than the practical threshold."
    if sharpe >= 1:
        threshold = 681
        if bars >= threshold:
            return "pass", "Sharpe >= 1 and bars >= 681", "Meets the practical threshold for confidence that true Sharpe is at least 0."
        return "review", "Sharpe >= 1 and bars >= 681", "Backtest Sharpe is adequate, but the sample is shorter than the practical threshold."
    return "fail", "Sharpe >= 1 with 681 bars, or Sharpe >= 2 with 174 bars", "Backtest Sharpe is below the practical confidence threshold for true Sharpe >= 0."


def _sample_size_true_sharpe_ge_one(sharpe: float | None, bars: int) -> tuple[str, str, str]:
    threshold = 2739
    if sharpe is None or not np.isfinite(sharpe):
        return "review", "Sharpe >= 1.5 and bars >= 2739", "Sharpe is unavailable, so true Sharpe >= 1 confidence cannot be evaluated."
    if sharpe >= 1.5 and bars >= threshold:
        return "pass", "Sharpe >= 1.5 and bars >= 2739", "Meets the practical threshold for confidence that true Sharpe is at least 1."
    if sharpe >= 1.5:
        return "review", "Sharpe >= 1.5 and bars >= 2739", "Backtest Sharpe is high enough, but the sample is shorter than the practical threshold."
    return "fail", "Sharpe >= 1.5 and bars >= 2739", "Backtest Sharpe is below the practical confidence threshold for true Sharpe >= 1."


def _coerce_float(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
