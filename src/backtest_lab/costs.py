from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CostModel:
    name: str
    commission_rate: float
    min_commission: float
    notes: str


COMMISSION_MODELS = {
    "pct_only": CostModel("pct_only", 0.0010, 0.0, "Simple percent commission from strategy YAML."),
    "us_equity_basic": CostModel("us_equity_basic", 0.0005, 1.0, "Research estimate for US equities; broker schedules differ."),
    "a_share_basic": CostModel("a_share_basic", 0.00055, 5.0, "Research estimate with commission plus sell-side stamp duty approximation."),
    "hk_equity_basic": CostModel("hk_equity_basic", 0.0030, 0.0, "Research estimate for HK commission and duties."),
    "crypto_basic": CostModel("crypto_basic", 0.0010, 0.0, "Research estimate for spot crypto taker fee."),
}

SLIPPAGE_MODELS = {
    "close": "Execute at close with no extra slippage assumption.",
    "hl_mean": "Use the high-low midpoint as a conservative reference price.",
    "hl_mean_gap_guard": "Use high-low midpoint and flag unusually large gaps for review.",
}


def resolve_commission_rate(model_name: str, fallback_rate: float) -> float:
    model = COMMISSION_MODELS.get(model_name)
    if model is None or model_name == "pct_only":
        return float(fallback_rate)
    return float(model.commission_rate)


def estimate_commission(model_name: str, trade_value: float, side: str = "buy") -> float:
    model = COMMISSION_MODELS.get(model_name, COMMISSION_MODELS["pct_only"])
    commission = abs(float(trade_value)) * model.commission_rate
    if side == "sell" and model_name == "a_share_basic":
        commission += abs(float(trade_value)) * 0.0005
    return max(commission, model.min_commission if trade_value else 0.0)


def describe_execution_assumptions(
    commission_model: str,
    slippage_model: str,
    position_model: str,
    benchmark: str,
) -> pd.DataFrame:
    cost_model = COMMISSION_MODELS.get(commission_model, COMMISSION_MODELS["pct_only"])
    return pd.DataFrame(
        [
            {"setting": "commission_model", "value": commission_model, "description": cost_model.notes},
            {"setting": "slippage_model", "value": slippage_model, "description": SLIPPAGE_MODELS.get(slippage_model, "Unknown slippage model")},
            {"setting": "position_model", "value": position_model, "description": "No leverage; capped by strategy position_size."},
            {"setting": "benchmark", "value": benchmark or "asset buy & hold", "description": "Used for ABU-style comparison metrics when available."},
        ]
    )


def slippage_reference(ohlcv: pd.DataFrame, model_name: str = "hl_mean_gap_guard", gap_threshold: float = 0.08) -> pd.DataFrame:
    data = ohlcv.copy()
    if data.empty:
        return data
    data["prev_close"] = data["close"].shift(1)
    if model_name in {"hl_mean", "hl_mean_gap_guard"}:
        data["execution_reference"] = (data["high"] + data["low"]) / 2
    else:
        data["execution_reference"] = data["close"]
    data["gap_pct"] = data["open"] / data["prev_close"] - 1
    data["gap_guard_flag"] = False
    if model_name == "hl_mean_gap_guard":
        data["gap_guard_flag"] = data["gap_pct"].abs() > float(gap_threshold)
    data["slippage_bps"] = np.where(
        data["close"].abs() > 0,
        (data["execution_reference"] / data["close"] - 1).abs() * 10000,
        0,
    )
    return data
