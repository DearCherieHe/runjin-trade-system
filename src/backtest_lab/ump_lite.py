from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.backtest_lab.costs import slippage_reference


@dataclass(frozen=True)
class UmpVerdict:
    verdict: str
    reasons: list[str]


def evaluate_ump_lite(ohlcv: pd.DataFrame, equity_curve: pd.DataFrame, trades: pd.DataFrame, config: dict | None = None) -> UmpVerdict:
    config = config or {}
    if config.get("enabled", True) is False:
        return UmpVerdict("allow", ["UMP-lite disabled by strategy spec"])

    reasons: list[str] = []
    max_volatility = float(config.get("max_volatility", 0.75))
    max_drawdown = float(config.get("max_drawdown", 0.12))
    min_avg_volume = float(config.get("min_avg_volume", 0))
    max_gap_pct = float(config.get("max_gap_pct", 0.10))
    max_consecutive_losses = int(config.get("max_consecutive_losses", 3))

    returns = pd.to_numeric(ohlcv.get("close", pd.Series(dtype=float)), errors="coerce").pct_change()
    realized_vol = float(returns.tail(20).std() * np.sqrt(252)) if returns.notna().sum() > 2 else 0.0
    if realized_vol > max_volatility:
        reasons.append(f"High volatility: {realized_vol:.1%} > {max_volatility:.1%}")

    if "Equity" in equity_curve.columns and not equity_curve.empty:
        equity = pd.to_numeric(equity_curve["Equity"], errors="coerce")
        drawdown = (equity / equity.cummax() - 1).min()
        if drawdown <= -max_drawdown:
            reasons.append(f"Max drawdown breach: {drawdown:.1%} <= -{max_drawdown:.1%}")

    if min_avg_volume > 0 and "volume" in ohlcv.columns:
        avg_volume = pd.to_numeric(ohlcv["volume"], errors="coerce").tail(20).mean()
        if pd.notna(avg_volume) and avg_volume < min_avg_volume:
            reasons.append(f"Low liquidity: avg volume {avg_volume:,.0f} < {min_avg_volume:,.0f}")

    slipped = slippage_reference(ohlcv, "hl_mean_gap_guard", max_gap_pct)
    if not slipped.empty and bool(slipped["gap_guard_flag"].tail(20).any()):
        reasons.append(f"Abnormal gap: recent gap exceeded {max_gap_pct:.1%}")

    consecutive_losses = _max_consecutive_losses(trades)
    if consecutive_losses >= max_consecutive_losses:
        reasons.append(f"Consecutive losses: {consecutive_losses} >= {max_consecutive_losses}")

    if len(reasons) >= 2:
        return UmpVerdict("block", reasons)
    if reasons:
        return UmpVerdict("review", reasons)
    return UmpVerdict("allow", ["No UMP-lite risk block triggered"])


def _max_consecutive_losses(trades: pd.DataFrame) -> int:
    if trades is None or trades.empty:
        return 0
    pnl = pd.to_numeric(trades.get("PnL", trades.get("ReturnPct", pd.Series(dtype=float))), errors="coerce").dropna()
    max_run = run = 0
    for value in pnl:
        if value < 0:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 0
    return max_run


def verdict_frame(verdict: UmpVerdict) -> pd.DataFrame:
    return pd.DataFrame({"verdict": [verdict.verdict] * len(verdict.reasons), "reason": verdict.reasons})
