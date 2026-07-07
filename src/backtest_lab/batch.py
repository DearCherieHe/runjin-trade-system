from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np
import pandas as pd


BATCH_STRATEGY_GRIDS: dict[str, dict[str, list[Any]]] = {
    "sma_crossover": {
        "fast_window": [10, 20, 50],
        "slow_window": [50, 100, 200],
    },
    "rsi_mean_reversion": {
        "rsi_window": [14],
        "lower_rsi": [30, 35, 40],
        "upper_rsi": [60, 65, 70],
    },
    "macd_trend": {
        "fast": [8, 12],
        "slow": [21, 26],
        "signal": [9],
    },
    "bollinger_reversion": {
        "window": [20, 30],
        "std": [2.0, 2.5],
    },
}


@dataclass
class BatchBacktestResult:
    leaderboard: pd.DataFrame
    equity_curves: pd.DataFrame
    warnings: list[str]


def build_parameter_grid(strategies: list[str] | None = None, max_variants_per_strategy: int = 8) -> list[dict[str, Any]]:
    selected = strategies or list(BATCH_STRATEGY_GRIDS)
    specs: list[dict[str, Any]] = []
    for strategy in selected:
        if strategy not in BATCH_STRATEGY_GRIDS:
            continue
        grid = BATCH_STRATEGY_GRIDS[strategy]
        keys = list(grid)
        variants = [dict(zip(keys, values)) for values in product(*(grid[key] for key in keys))]
        for params in variants[: max(1, int(max_variants_per_strategy))]:
            specs.append({"strategy": strategy, "params": params})
    return specs


def run_batch_backtest(
    raw_prices: pd.DataFrame,
    tickers: list[str] | None = None,
    strategies: list[str] | None = None,
    max_tickers: int = 30,
    max_variants_per_strategy: int = 8,
    cash: float = 100000,
    commission_pct: float = 0.10,
    min_bars: int = 80,
) -> BatchBacktestResult:
    prices = _prepare_close_matrix(raw_prices, tickers, max_tickers)
    warnings: list[str] = []
    if prices.empty:
        return BatchBacktestResult(pd.DataFrame(), pd.DataFrame(), ["No usable price history for batch backtest."])

    specs = build_parameter_grid(strategies, max_variants_per_strategy)
    if not specs:
        return BatchBacktestResult(pd.DataFrame(), pd.DataFrame(), ["No supported strategy templates selected."])

    rows = []
    equity_frames = []
    commission = max(0.0, float(commission_pct) / 100)
    for ticker in prices.columns:
        close = prices[ticker].dropna()
        if len(close) < min_bars:
            warnings.append(f"{ticker} skipped: only {len(close)} bars, needs at least {min_bars}.")
            continue
        for spec in specs:
            signals = _build_signals(close, spec["strategy"], spec["params"])
            metrics, curve = _simulate_long_only(close, signals["entry"], signals["exit"], cash, commission)
            params_text = _format_params(spec["params"])
            rows.append(
                {
                    "ticker": ticker,
                    "strategy": spec["strategy"],
                    "params": params_text,
                    "return_pct": metrics["return_pct"],
                    "max_drawdown_pct": metrics["max_drawdown_pct"],
                    "sharpe": metrics["sharpe"],
                    "win_rate_pct": metrics["win_rate_pct"],
                    "trades": metrics["trades"],
                    "score": metrics["score"],
                }
            )
            curve = curve.assign(ticker=ticker, strategy=spec["strategy"], params=params_text)
            equity_frames.append(curve)

    leaderboard = pd.DataFrame(rows)
    if leaderboard.empty:
        return BatchBacktestResult(leaderboard, pd.DataFrame(), warnings or ["No strategies produced results."])

    leaderboard = leaderboard.sort_values(["score", "return_pct"], ascending=[False, False]).reset_index(drop=True)
    equity_curves = pd.concat(equity_frames, ignore_index=True) if equity_frames else pd.DataFrame()
    return BatchBacktestResult(leaderboard, equity_curves, warnings)


def _prepare_close_matrix(raw_prices: pd.DataFrame, tickers: list[str] | None, max_tickers: int) -> pd.DataFrame:
    required = {"date", "ticker", "close"}
    missing = required - set(raw_prices.columns)
    if missing:
        raise ValueError(f"Missing price columns: {', '.join(sorted(missing))}")

    data = raw_prices[["date", "ticker", "close"]].copy()
    data["ticker"] = data["ticker"].astype(str).str.upper()
    if tickers:
        selected = [str(ticker).upper() for ticker in tickers if str(ticker).strip()]
        data = data.loc[data["ticker"].isin(selected)]
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["close"] = pd.to_numeric(data["close"], errors="coerce")
    matrix = data.dropna().pivot_table(index="date", columns="ticker", values="close", aggfunc="last").sort_index()
    if matrix.empty:
        return matrix
    coverage = matrix.notna().sum().sort_values(ascending=False)
    selected_cols = coverage.head(max(1, int(max_tickers))).index.tolist()
    return matrix[selected_cols].ffill().dropna(axis=1, how="all")


def _build_signals(close: pd.Series, strategy: str, params: dict[str, Any]) -> dict[str, pd.Series]:
    if strategy == "sma_crossover":
        fast = close.rolling(int(params.get("fast_window", 20))).mean()
        slow = close.rolling(int(params.get("slow_window", 60))).mean()
        entry = (fast > slow) & (fast.shift(1) <= slow.shift(1))
        exit_signal = (fast < slow) & (fast.shift(1) >= slow.shift(1))
    elif strategy == "rsi_mean_reversion":
        rsi = _rsi(close, int(params.get("rsi_window", 14)))
        entry = rsi < float(params.get("lower_rsi", 35))
        exit_signal = rsi > float(params.get("upper_rsi", 65))
    elif strategy == "macd_trend":
        macd, signal = _macd(close, int(params.get("fast", 12)), int(params.get("slow", 26)), int(params.get("signal", 9)))
        entry = (macd > signal) & (macd.shift(1) <= signal.shift(1))
        exit_signal = (macd < signal) & (macd.shift(1) >= signal.shift(1))
    elif strategy == "bollinger_reversion":
        window = int(params.get("window", 20))
        mid = close.rolling(window).mean()
        std = close.rolling(window).std()
        lower = mid - float(params.get("std", 2.0)) * std
        entry = close < lower
        exit_signal = close > mid
    else:
        entry = pd.Series(False, index=close.index)
        exit_signal = pd.Series(False, index=close.index)
    return {"entry": entry.fillna(False), "exit": exit_signal.fillna(False)}


def _simulate_long_only(close: pd.Series, entry: pd.Series, exit_signal: pd.Series, cash: float, commission: float) -> tuple[dict[str, float], pd.DataFrame]:
    position = []
    in_position = False
    entry_prices = []
    trade_returns = []
    current_entry = None
    for date, price in close.items():
        if not in_position and bool(entry.loc[date]):
            in_position = True
            current_entry = float(price)
            entry_prices.append(current_entry)
        elif in_position and bool(exit_signal.loc[date]):
            in_position = False
            if current_entry and current_entry > 0:
                trade_returns.append(float(price) / current_entry - 1 - commission * 2)
            current_entry = None
        position.append(1.0 if in_position else 0.0)

    pos = pd.Series(position, index=close.index)
    returns = close.pct_change().fillna(0)
    turnover = pos.diff().abs().fillna(pos.abs())
    strategy_returns = pos.shift(1).fillna(0) * returns - turnover * commission
    equity = cash * (1 + strategy_returns).cumprod()
    drawdown = equity / equity.cummax() - 1
    daily = strategy_returns.dropna()
    total_return = equity.iloc[-1] / cash - 1 if len(equity) else 0
    sharpe = daily.mean() / daily.std() * np.sqrt(252) if len(daily) > 2 and daily.std() > 0 else np.nan
    completed_trades = len(trade_returns)
    win_rate = sum(ret > 0 for ret in trade_returns) / completed_trades if completed_trades else 0.0
    max_drawdown = float(drawdown.min()) if not drawdown.empty else 0.0
    score = total_return * 100 + (float(sharpe) if pd.notna(sharpe) else 0) * 4 + max_drawdown * 45
    curve = pd.DataFrame(
        {
            "date": close.index,
            "equity": equity.values,
            "drawdown_pct": drawdown.values * 100,
            "position": pos.values,
        }
    )
    return (
        {
            "return_pct": float(total_return * 100),
            "max_drawdown_pct": float(max_drawdown * 100),
            "sharpe": float(sharpe) if pd.notna(sharpe) else np.nan,
            "win_rate_pct": float(win_rate * 100),
            "trades": int(len(entry_prices)),
            "score": float(score),
        },
        curve,
    )


def _rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = -delta.clip(upper=0).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd(close: pd.Series, fast: int, slow: int, signal: int) -> tuple[pd.Series, pd.Series]:
    macd = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    return macd, macd.ewm(span=signal, adjust=False).mean()


def _format_params(params: dict[str, Any]) -> str:
    return ", ".join(f"{key}={value}" for key, value in params.items())
