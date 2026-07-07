from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.backtest_lab.bias import build_position_file


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


def out_of_sample_audit_frame(data: pd.DataFrame, spec: dict[str, Any], full_stats: dict[str, Any]) -> pd.DataFrame:
    config = spec.get("out_of_sample_check", {}) or {}
    split_pct = float(config.get("train_pct", 0.70))
    min_test_bars = int(config.get("min_test_bars", 174))
    max_sharpe_decay = float(config.get("max_sharpe_decay", 0.75))
    if data.empty:
        return pd.DataFrame(
            [
                {
                    "check": "out_of_sample_split",
                    "status": "review",
                    "train_value": "N/A",
                    "test_value": "N/A",
                    "threshold": "non-empty data",
                    "reason": "No OHLCV rows available for out-of-sample testing.",
                }
            ]
        )
    split_idx = int(len(data) * split_pct)
    split_idx = min(max(split_idx, 20), len(data) - 1)
    train = data.iloc[:split_idx].copy()
    test = data.iloc[split_idx:].copy()
    template = spec["template"]
    params = spec.get("parameters", {}) or {}
    trade_on_close = bool(spec.get("trade_on_close", False))
    train_metrics = _segment_metrics(train, template, params, trade_on_close)
    test_metrics = _segment_metrics(test, template, params, trade_on_close)
    full_sharpe = _coerce_float(full_stats.get("Sharpe Ratio"))
    sharpe_decay = None
    if train_metrics["sharpe"] is not None and test_metrics["sharpe"] is not None:
        sharpe_decay = train_metrics["sharpe"] - test_metrics["sharpe"]
    sample_status, sample_threshold, sample_reason = _sample_size_true_sharpe_ge_zero(test_metrics["sharpe"], len(test))
    rows = [
        {
            "check": "out_of_sample_split",
            "status": "pass" if 0.5 <= split_pct <= 0.85 and len(test) > 0 else "review",
            "train_value": f"{len(train)} bars",
            "test_value": f"{len(test)} bars",
            "threshold": "50%-85% train with non-empty test",
            "reason": "Train segment is used for model design; test segment is reserved for unseen historical validation.",
        },
        {
            "check": "oos_min_test_bars",
            "status": "pass" if len(test) >= min_test_bars else "review",
            "train_value": len(train),
            "test_value": len(test),
            "threshold": f">= {min_test_bars} test bars",
            "reason": "Out-of-sample and paper-trading evidence needs enough independent observations.",
        },
        {
            "check": "oos_return_reasonable",
            "status": "pass" if test_metrics["return_pct"] >= -10 else "review" if test_metrics["return_pct"] >= -25 else "fail",
            "train_value": round(train_metrics["return_pct"], 2),
            "test_value": round(test_metrics["return_pct"], 2),
            "threshold": "test return > -10% preferred",
            "reason": "A strategy optimized in-sample should not collapse in the reserved test segment.",
        },
        {
            "check": "oos_sharpe_decay",
            "status": _status_oos_sharpe_decay(train_metrics["sharpe"], test_metrics["sharpe"], max_sharpe_decay),
            "train_value": round(train_metrics["sharpe"], 3) if train_metrics["sharpe"] is not None else "N/A",
            "test_value": round(test_metrics["sharpe"], 3) if test_metrics["sharpe"] is not None else "N/A",
            "threshold": f"test sharpe decay <= {max_sharpe_decay}",
            "reason": f"Full-period Sharpe={round(full_sharpe, 3) if full_sharpe is not None else 'N/A'}; train-test Sharpe decay={round(sharpe_decay, 3) if sharpe_decay is not None else 'N/A'}.",
        },
        {
            "check": "oos_sample_size_true_sharpe_ge_0",
            "status": sample_status,
            "train_value": "N/A",
            "test_value": f"bars={len(test)}, sharpe={round(test_metrics['sharpe'], 3) if test_metrics['sharpe'] is not None else 'N/A'}",
            "threshold": sample_threshold,
            "reason": sample_reason,
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


def _segment_metrics(data: pd.DataFrame, template: str, params: dict[str, Any], trade_on_close: bool) -> dict[str, float | None]:
    if len(data) < 5:
        return {"return_pct": 0.0, "sharpe": None, "max_drawdown_pct": 0.0, "exposure_pct": 0.0}
    positions = build_position_file(data, template, params, trade_on_close).set_index("date")
    close = data["Close"].copy()
    returns = close.pct_change().fillna(0)
    executable_position = positions["executable_position"].reindex(data.index).fillna(0)
    strategy_returns = executable_position * returns
    equity = (1 + strategy_returns).cumprod()
    drawdown = equity / equity.cummax() - 1
    daily = strategy_returns.dropna()
    sharpe = daily.mean() / daily.std() * np.sqrt(252) if len(daily) > 2 and daily.std() > 0 else None
    return {
        "return_pct": float((equity.iloc[-1] - 1) * 100) if len(equity) else 0.0,
        "sharpe": float(sharpe) if sharpe is not None and np.isfinite(sharpe) else None,
        "max_drawdown_pct": float(drawdown.min() * 100) if len(drawdown) else 0.0,
        "exposure_pct": float(executable_position.mean() * 100) if len(executable_position) else 0.0,
    }


def _status_oos_sharpe_decay(train_sharpe: float | None, test_sharpe: float | None, max_decay: float) -> str:
    if train_sharpe is None or test_sharpe is None:
        return "review"
    if test_sharpe < 0 <= train_sharpe:
        return "fail"
    if train_sharpe - test_sharpe <= max_decay:
        return "pass"
    return "review"


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
