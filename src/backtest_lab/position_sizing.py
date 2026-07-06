from __future__ import annotations

import numpy as np
import pandas as pd


def atr_series(ohlcv: pd.DataFrame, window: int = 21) -> pd.Series:
    high = pd.to_numeric(ohlcv["high"], errors="coerce")
    low = pd.to_numeric(ohlcv["low"], errors="coerce")
    close = pd.to_numeric(ohlcv["close"], errors="coerce")
    prev_close = close.shift(1).fillna(close)
    true_range = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return true_range.ewm(span=window, adjust=False, min_periods=1).mean()


def atr_position_size(
    ohlcv: pd.DataFrame,
    max_position_pct: float,
    atr_window: int = 21,
    risk_budget_pct: float = 0.015,
    atr_stop_multiple: float = 2.5,
) -> float:
    if ohlcv.empty:
        return min(float(max_position_pct), 1.0)
    atr = atr_series(ohlcv, atr_window).dropna()
    close = pd.to_numeric(ohlcv["close"], errors="coerce").dropna()
    if atr.empty or close.empty or close.iloc[-1] <= 0:
        return min(float(max_position_pct), 1.0)
    atr_pct = float(atr.iloc[-1] / close.iloc[-1])
    if atr_pct <= 0 or not np.isfinite(atr_pct):
        return min(float(max_position_pct), 1.0)
    risk_position = float(risk_budget_pct) / (atr_pct * float(atr_stop_multiple))
    return float(np.clip(risk_position, 0.01, float(max_position_pct)))


def capped_kelly_position_size(
    win_rate: float,
    avg_win_pct: float,
    avg_loss_pct: float,
    max_position_pct: float,
    fraction: float = 0.25,
) -> float:
    loss = abs(float(avg_loss_pct))
    if loss <= 0:
        return min(float(max_position_pct), 0.1)
    payoff = abs(float(avg_win_pct)) / loss
    if payoff <= 0:
        return min(float(max_position_pct), 0.1)
    raw_kelly = float(win_rate) - (1 - float(win_rate)) / payoff
    return float(np.clip(raw_kelly * float(fraction), 0.0, float(max_position_pct)))


def resolve_position_size(raw_ohlcv: pd.DataFrame, spec: dict, warnings: list[str]) -> float:
    position_model = spec.get("position_model", "fixed_fraction")
    max_position = float(spec.get("position_size", 0.95))
    if position_model == "fixed_fraction":
        return max_position
    if position_model == "atr_risk":
        params = spec.get("position_parameters", {}) or {}
        size = atr_position_size(
            raw_ohlcv,
            max_position,
            atr_window=int(params.get("atr_window", 21)),
            risk_budget_pct=float(params.get("risk_budget_pct", 0.015)),
            atr_stop_multiple=float(params.get("atr_stop_multiple", 2.5)),
        )
        warnings.append(f"ATR position sizing capped exposure at {size:.1%}.")
        return size
    if position_model == "kelly_lite":
        params = spec.get("position_parameters", {}) or {}
        size = capped_kelly_position_size(
            float(params.get("win_rate", 0.50)),
            float(params.get("avg_win_pct", 0.04)),
            float(params.get("avg_loss_pct", -0.03)),
            max_position,
            fraction=float(params.get("fraction", 0.25)),
        )
        warnings.append(f"Kelly-lite position sizing capped exposure at {size:.1%}; research only.")
        return size
    raise ValueError(f"Unsupported position_model: {position_model}")
