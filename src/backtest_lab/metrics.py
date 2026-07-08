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
    drawdown_duration = _drawdown_duration(equity_curve, drawdown)
    drawdown_depth = _drawdown_depth(equity_curve, drawdown)
    years = len(equity) / 252 if len(equity) else 0

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
        "max_drawdown_start": drawdown_depth["start"],
        "max_drawdown_trough": drawdown_depth["trough"],
        "max_drawdown_duration_bars": drawdown_duration["bars"],
        "max_drawdown_duration_days": drawdown_duration["days"],
        "max_drawdown_duration_start": drawdown_duration["start"],
        "max_drawdown_duration_end": drawdown_duration["end"],
        "trades_per_year": float(len(trades) / years) if years > 0 and trades is not None else 0,
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


def _drawdown_duration(equity_curve: pd.DataFrame, drawdown: pd.Series) -> dict:
    if drawdown.empty:
        return {"bars": 0, "days": 0, "start": None, "end": None}
    max_bars = 0
    current_bars = 0
    start_idx = None
    best_start = None
    best_end = None
    for idx, value in drawdown.items():
        if value < 0:
            if current_bars == 0:
                start_idx = idx
            current_bars += 1
            if current_bars > max_bars:
                max_bars = current_bars
                best_start = start_idx
                best_end = idx
        else:
            current_bars = 0
            start_idx = None
    days = max_bars
    start_date = None
    end_date = None
    date_col = "Date" if "Date" in equity_curve.columns else "date" if "date" in equity_curve.columns else None
    if date_col and best_start is not None and best_end is not None:
        dates = pd.to_datetime(equity_curve[date_col], errors="coerce")
        if best_start in dates.index and best_end in dates.index and pd.notna(dates.loc[best_start]) and pd.notna(dates.loc[best_end]):
            days = max(0, int((dates.loc[best_end] - dates.loc[best_start]).days))
            start_date = dates.loc[best_start]
            end_date = dates.loc[best_end]
    return {"bars": int(max_bars), "days": int(days), "start": start_date, "end": end_date}


def _drawdown_depth(equity_curve: pd.DataFrame, drawdown: pd.Series) -> dict:
    if drawdown.empty:
        return {"start": None, "trough": None}
    trough_idx = drawdown.idxmin()
    equity_col = "Equity" if "Equity" in equity_curve.columns else equity_curve.columns[-1]
    equity = pd.to_numeric(equity_curve[equity_col], errors="coerce")
    peak_idx = equity.loc[:trough_idx].idxmax() if trough_idx in equity.index else None
    date_col = "Date" if "Date" in equity_curve.columns else "date" if "date" in equity_curve.columns else None
    if not date_col:
        return {"start": peak_idx, "trough": trough_idx}
    dates = pd.to_datetime(equity_curve[date_col], errors="coerce")
    start = dates.loc[peak_idx] if peak_idx in dates.index and pd.notna(dates.loc[peak_idx]) else None
    trough = dates.loc[trough_idx] if trough_idx in dates.index and pd.notna(dates.loc[trough_idx]) else None
    return {"start": start, "trough": trough}


def drawdown_tolerance_frame(metrics: dict, spec: dict) -> pd.DataFrame:
    config = spec.get("drawdown_tolerance", {}) or {}
    max_depth_pct = float(config.get("max_drawdown_pct", 10))
    max_duration_days = int(config.get("max_drawdown_duration_days", 90))
    max_duration_bars = int(config.get("max_drawdown_duration_bars", 63))
    actual_depth = abs(float(metrics.get("max_drawdown_pct") or 0))
    actual_days = int(metrics.get("max_drawdown_duration_days") or 0)
    actual_bars = int(metrics.get("max_drawdown_duration_bars") or 0)
    return pd.DataFrame(
        [
            {
                "check": "max_drawdown_depth_tolerance",
                "status": "pass" if actual_depth <= max_depth_pct else "fail",
                "actual": f"{actual_depth:.1f}%",
                "tolerance": f"<= {max_depth_pct:.1f}%",
                "period": f"{metrics.get('max_drawdown_start') or 'N/A'} -> {metrics.get('max_drawdown_trough') or 'N/A'}",
                "reason": "Maximum peak-to-trough loss must stay inside your liquidation/strategy-stop threshold.",
            },
            {
                "check": "max_drawdown_duration_days_tolerance",
                "status": "pass" if actual_days <= max_duration_days else "fail",
                "actual": f"{actual_days} days",
                "tolerance": f"<= {max_duration_days} days",
                "period": f"{metrics.get('max_drawdown_duration_start') or 'N/A'} -> {metrics.get('max_drawdown_duration_end') or 'N/A'}",
                "reason": "Longest time underwater must stay inside your psychological and capital patience threshold.",
            },
            {
                "check": "max_drawdown_duration_bars_tolerance",
                "status": "pass" if actual_bars <= max_duration_bars else "review",
                "actual": f"{actual_bars} bars",
                "tolerance": f"<= {max_duration_bars} bars",
                "period": f"{metrics.get('max_drawdown_duration_start') or 'N/A'} -> {metrics.get('max_drawdown_duration_end') or 'N/A'}",
                "reason": "Bar-based duration catches long underwater periods even when calendar day spacing changes by timeframe.",
            },
        ]
    )


def metrics_detail_frame(metrics: dict) -> pd.DataFrame:
    rows = []
    for key, value in metrics.items():
        rows.append({"metric": key, "value": value})
    return pd.DataFrame(rows)
