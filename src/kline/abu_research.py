from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest_lab.position_sizing import atr_series


def gap_analysis(ohlcv: pd.DataFrame, window: int = 21, gap_factor: float = 1.0) -> pd.DataFrame:
    data = ohlcv.copy()
    if data.empty:
        return pd.DataFrame(columns=["date", "direction", "gap_pct", "gap_power", "volume_ratio"])
    data["prev_close"] = data["close"].shift(1)
    data["abs_return"] = data["close"].pct_change().abs()
    data["avg_abs_return"] = data["abs_return"].rolling(window, min_periods=3).mean()
    data["avg_volume"] = data["volume"].rolling(window, min_periods=3).mean()
    up_gap = (data["low"] > data["prev_close"]) & (data["abs_return"] > data["avg_abs_return"] * gap_factor)
    down_gap = (data["high"] < data["prev_close"]) & (data["abs_return"] > data["avg_abs_return"] * gap_factor)
    gaps = data.loc[up_gap | down_gap].copy()
    if gaps.empty:
        return pd.DataFrame(columns=["date", "direction", "gap_pct", "gap_power", "volume_ratio"])
    gaps["direction"] = np.where(gaps["low"] > gaps["prev_close"], "gap_up", "gap_down")
    gaps["gap_pct"] = np.where(
        gaps["direction"] == "gap_up",
        gaps["low"] / gaps["prev_close"] - 1,
        gaps["high"] / gaps["prev_close"] - 1,
    )
    gaps["gap_power"] = gaps["gap_pct"].abs() / gaps["avg_abs_return"].replace(0, np.nan)
    gaps["volume_ratio"] = gaps["volume"] / gaps["avg_volume"].replace(0, np.nan)
    return gaps[["date", "direction", "gap_pct", "gap_power", "volume_ratio"]].dropna().reset_index(drop=True)


def atr_research(ohlcv: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    data = ohlcv[["date", "close"]].copy()
    data[f"atr{window}"] = atr_series(ohlcv, window)
    data["atr_pct"] = data[f"atr{window}"] / data["close"].replace(0, np.nan)
    return data


def rolling_correlation_matrix(price_df: pd.DataFrame, tickers: list[str], window: int = 60) -> pd.DataFrame:
    if price_df.empty or len(tickers) < 2:
        return pd.DataFrame()
    data = price_df.loc[price_df["ticker"].isin(tickers), ["date", "ticker", "close"]].copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    matrix = data.dropna().pivot_table(index="date", columns="ticker", values="close", aggfunc="last").sort_index().ffill()
    returns = matrix.pct_change().dropna().tail(window)
    if returns.empty:
        return pd.DataFrame()
    return returns.corr()


def similar_paths(price_df: pd.DataFrame, ticker: str, candidates: list[str], window: int = 60) -> pd.DataFrame:
    if price_df.empty:
        return pd.DataFrame(columns=["ticker", "similarity"])
    symbols = [ticker] + [item for item in candidates if item != ticker]
    data = price_df.loc[price_df["ticker"].isin(symbols), ["date", "ticker", "close"]].copy()
    matrix = data.pivot_table(index="date", columns="ticker", values="close", aggfunc="last").sort_index().ffill().tail(window)
    if ticker not in matrix or matrix.shape[0] < 5:
        return pd.DataFrame(columns=["ticker", "similarity"])
    norm = matrix / matrix.iloc[0] - 1
    base = norm[ticker]
    rows = []
    for symbol in norm.columns:
        if symbol == ticker:
            continue
        corr = base.corr(norm[symbol])
        if pd.notna(corr):
            rows.append({"ticker": symbol, "similarity": float(corr)})
    return pd.DataFrame(rows).sort_values("similarity", ascending=False).reset_index(drop=True)
