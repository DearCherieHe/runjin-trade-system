from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class LookaheadAuditResult:
    status: str
    compared_rows: int
    mismatches: int
    truncation_bars: int
    details: pd.DataFrame


def build_position_file(data: pd.DataFrame, template: str, params: dict[str, Any], trade_on_close: bool = False) -> pd.DataFrame:
    close = data["Close"].copy()
    if template == "sma_crossover":
        fast = close.rolling(int(params.get("fast_window", 20))).mean()
        slow = close.rolling(int(params.get("slow_window", 60))).mean()
        entry = (fast > slow) & (fast.shift(1) <= slow.shift(1))
        exit_signal = (fast < slow) & (fast.shift(1) >= slow.shift(1))
    elif template == "rsi_mean_reversion":
        rsi = _rsi(close, int(params.get("rsi_window", 14)))
        entry = rsi < float(params.get("lower_rsi", 35))
        exit_signal = rsi > float(params.get("upper_rsi", 65))
    elif template == "bollinger_reversion":
        window = int(params.get("window", 20))
        mid = close.rolling(window).mean()
        std = close.rolling(window).std()
        lower = mid - float(params.get("std", 2.0)) * std
        entry = close < lower
        exit_signal = close > mid
    elif template == "macd_trend":
        macd, signal = _macd(close, int(params.get("fast", 12)), int(params.get("slow", 26)), int(params.get("signal", 9)))
        entry = (macd > signal) & (macd.shift(1) <= signal.shift(1))
        exit_signal = (macd < signal) & (macd.shift(1) >= signal.shift(1))
    else:
        entry = pd.Series(False, index=close.index)
        exit_signal = pd.Series(False, index=close.index)

    signal_position = _signals_to_position(entry.fillna(False), exit_signal.fillna(False))
    executable_position = signal_position if trade_on_close else signal_position.shift(1).fillna(0)
    return pd.DataFrame(
        {
            "date": data.index,
            "entry_signal": entry.fillna(False).astype(int).to_numpy(),
            "exit_signal": exit_signal.fillna(False).astype(int).to_numpy(),
            "signal_position": signal_position.to_numpy(),
            "executable_position": executable_position.to_numpy(),
        }
    )


def run_lookahead_audit(
    data: pd.DataFrame,
    template: str,
    params: dict[str, Any],
    trade_on_close: bool = False,
    truncation_bars: int = 30,
) -> LookaheadAuditResult:
    if data.empty:
        return LookaheadAuditResult("review", 0, 0, truncation_bars, pd.DataFrame([{"reason": "No data to audit"}]))
    n = max(1, int(truncation_bars))
    if len(data) <= n + 5:
        return LookaheadAuditResult(
            "review",
            0,
            0,
            n,
            pd.DataFrame([{"reason": f"Not enough rows for truncation audit: rows={len(data)}, truncation={n}"}]),
        )

    full_positions = build_position_file(data, template, params, trade_on_close)
    truncated_positions = build_position_file(data.iloc[:-n].copy(), template, params, trade_on_close)
    full_comparable = full_positions.iloc[:-n].reset_index(drop=True)
    truncated_comparable = truncated_positions.reset_index(drop=True)
    compare_cols = ["entry_signal", "exit_signal", "signal_position", "executable_position"]
    length = min(len(full_comparable), len(truncated_comparable))
    full_comparable = full_comparable.iloc[:length].copy()
    truncated_comparable = truncated_comparable.iloc[:length].copy()
    mismatches = full_comparable[compare_cols].ne(truncated_comparable[compare_cols]).any(axis=1)
    mismatch_rows = full_comparable.loc[mismatches, ["date"] + compare_cols].copy()
    if not mismatch_rows.empty:
        mismatch_rows = mismatch_rows.assign(reason="Position changed after future rows were removed")
    status = "pass" if int(mismatches.sum()) == 0 else "fail"
    if status == "pass":
        details = pd.DataFrame(
            [
                {
                    "reason": "Full-data positions match truncated-data positions through the shared history.",
                    "last_compared_date": full_comparable["date"].iloc[-1],
                }
            ]
        )
    else:
        details = mismatch_rows.head(20)
    return LookaheadAuditResult(status, int(length), int(mismatches.sum()), n, details)


def lookahead_audit_frame(result: LookaheadAuditResult) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "check": "truncated_position_consistency",
                "status": result.status,
                "compared_rows": result.compared_rows,
                "mismatches": result.mismatches,
                "truncation_bars": result.truncation_bars,
            }
        ]
    )


def _signals_to_position(entry: pd.Series, exit_signal: pd.Series) -> pd.Series:
    position = []
    in_position = False
    for date in entry.index:
        if not in_position and bool(entry.loc[date]):
            in_position = True
        elif in_position and bool(exit_signal.loc[date]):
            in_position = False
        position.append(1.0 if in_position else 0.0)
    return pd.Series(position, index=entry.index)


def _rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = -delta.clip(upper=0).rolling(window).mean()
    rs = gain / loss.replace(0, float("nan"))
    return 100 - (100 / (1 + rs))


def _macd(close: pd.Series, fast: int, slow: int, signal: int) -> tuple[pd.Series, pd.Series]:
    macd = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    return macd, macd.ewm(span=signal, adjust=False).mean()
