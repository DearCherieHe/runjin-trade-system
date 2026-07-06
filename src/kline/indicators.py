import numpy as np
import pandas as pd


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = -delta.clip(upper=0).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def kdj(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 9) -> tuple[pd.Series, pd.Series, pd.Series]:
    lowest_low = low.rolling(window).min()
    highest_high = high.rolling(window).max()
    rsv = (close - lowest_low) / (highest_high - lowest_low).replace(0, np.nan) * 100
    k = rsv.ewm(com=2, adjust=False).mean()
    d = k.ewm(com=2, adjust=False).mean()
    j = 3 * k - 2 * d
    return k, d, j


def add_indicators(df: pd.DataFrame, price_col: str = "close") -> pd.DataFrame:
    data = df.copy()
    close = data[price_col]
    data["ma20"] = close.rolling(20).mean()
    data["ma60"] = close.rolling(60).mean()
    data["return"] = close.pct_change()
    data["volatility_20"] = data["return"].rolling(20).std() * np.sqrt(252)
    data["rsi14"] = rsi(close, 14)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    data["macd"] = ema12 - ema26
    data["macd_signal"] = data["macd"].ewm(span=9, adjust=False).mean()

    mid = close.rolling(20).mean()
    std = close.rolling(20).std()
    data["bb_mid"] = mid
    data["bb_upper"] = mid + 2 * std
    data["bb_lower"] = mid - 2 * std

    if {"high", "low", price_col}.issubset(data.columns):
        data["kdj_k"], data["kdj_d"], data["kdj_j"] = kdj(data["high"], data["low"], close)
    return data


def classify_regime(indicator_df: pd.DataFrame) -> str:
    recent = indicator_df.dropna().tail(1)
    if recent.empty:
        return "Insufficient data"
    row = recent.iloc[0]
    if row["close"] > row["ma20"] > row["ma60"] and row["volatility_20"] < 0.75:
        return "Uptrend with controlled volatility"
    if row["close"] < row["ma20"] < row["ma60"]:
        return "Downtrend / avoid adding"
    if row["rsi14"] < 35:
        return "Oversold mean-reversion watch"
    if row["rsi14"] > 70:
        return "Extended / chase risk"
    return "Range or transition"


def relative_strength(ticker_df: pd.DataFrame, benchmark_df: pd.DataFrame) -> pd.DataFrame:
    left = ticker_df[["date", "close"]].rename(columns={"close": "ticker_close"})
    right = benchmark_df[["date", "close"]].rename(columns={"close": "benchmark_close"})
    merged = left.merge(right, on="date", how="inner")
    merged["relative_strength"] = (
        merged["ticker_close"] / merged["ticker_close"].iloc[0]
    ) / (merged["benchmark_close"] / merged["benchmark_close"].iloc[0])
    return merged
