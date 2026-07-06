from __future__ import annotations

import numpy as np
import pandas as pd


def abu_style_metrics(equity_curve: pd.DataFrame, trades: pd.DataFrame, data: pd.DataFrame) -> dict:
    if equity_curve.empty:
        return {}
    equity_col = "Equity" if "Equity" in equity_curve.columns else equity_curve.columns[-1]
    equity = pd.to_numeric(equity_curve[equity_col], errors="coerce").dropna()
    returns = equity.pct_change().dropna()
    if equity.empty:
        return {}

    annualized_vol = float(returns.std() * np.sqrt(252)) if not returns.empty else 0.0
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1) if equity.iloc[0] else 0.0
    annualized_return = (1 + total_return) ** (252 / max(len(equity), 1)) - 1 if len(equity) > 1 else 0.0
    sharpe = annualized_return / annualized_vol if annualized_vol > 0 else np.nan
    drawdown = equity / equity.cummax() - 1

    benchmark_returns = pd.to_numeric(data.get("Close", data.get("close", pd.Series(dtype=float))), errors="coerce").pct_change().dropna()
    aligned = pd.concat([returns.reset_index(drop=True), benchmark_returns.reset_index(drop=True)], axis=1).dropna()
    alpha = beta = information_ratio = np.nan
    if aligned.shape[0] > 3:
        algo = aligned.iloc[:, 0]
        bench = aligned.iloc[:, 1]
        bench_var = bench.var()
        beta = float(algo.cov(bench) / bench_var) if bench_var else np.nan
        alpha = float((algo.mean() - (beta if pd.notna(beta) else 0) * bench.mean()) * 252)
        active = algo - bench
        information_ratio = float(active.mean() / active.std() * np.sqrt(252)) if active.std() else np.nan

    trade_metrics = _trade_metrics(trades)
    metrics = {
        "cash_utilization_proxy_pct": float(100 - equity_curve.get("Cash", pd.Series([np.nan])).mean() / equity.mean() * 100)
        if "Cash" in equity_curve.columns and equity.mean()
        else None,
        "annualized_return_pct": annualized_return * 100,
        "annualized_volatility_pct": annualized_vol * 100,
        "sharpe_ratio_abu": float(sharpe) if pd.notna(sharpe) else None,
        "alpha_annualized": alpha if pd.notna(alpha) else None,
        "beta": beta if pd.notna(beta) else None,
        "information_ratio": information_ratio if pd.notna(information_ratio) else None,
        "max_drawdown_pct": float(drawdown.min() * 100),
    }
    metrics.update(trade_metrics)
    return metrics


def _trade_metrics(trades: pd.DataFrame) -> dict:
    if trades is None or trades.empty:
        return {
            "avg_holding_days": 0,
            "profit_loss_ratio": 0,
            "buy_reason_distribution": "No trades",
            "sell_reason_distribution": "No trades",
        }
    returns = pd.to_numeric(trades.get("ReturnPct", pd.Series(dtype=float)), errors="coerce")
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    avg_win = wins.mean() if not wins.empty else 0
    avg_loss = abs(losses.mean()) if not losses.empty else 0
    if "Duration" in trades.columns:
        holding = pd.to_timedelta(trades["Duration"], errors="coerce").dt.total_seconds() / 86400
        avg_holding = float(holding.mean()) if holding.notna().any() else 0
    else:
        avg_holding = 0
    return {
        "avg_holding_days": avg_holding,
        "profit_loss_ratio": float(avg_win / avg_loss) if avg_loss else 0,
        "buy_reason_distribution": "Template entry signal",
        "sell_reason_distribution": "Template exit / stop / take-profit",
    }


def metrics_detail_frame(metrics: dict) -> pd.DataFrame:
    rows = []
    for key, value in metrics.items():
        rows.append({"metric": key, "value": value})
    return pd.DataFrame(rows)
