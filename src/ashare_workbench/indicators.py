import numpy as np
import pandas as pd


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = -delta.clip(upper=0).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _kdj(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 9) -> tuple[pd.Series, pd.Series, pd.Series]:
    lowest = low.rolling(window).min()
    highest = high.rolling(window).max()
    rsv = (close - lowest) / (highest - lowest).replace(0, np.nan) * 100
    k = rsv.ewm(com=2, adjust=False).mean()
    d = k.ewm(com=2, adjust=False).mean()
    return k, d, 3 * k - 2 * d


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(window).mean()


def ashare_limit_pct(ticker: str, name: str = "") -> float:
    code = str(ticker)
    label = str(name).upper()
    if "ST" in label:
        return 0.05
    if code.startswith(("300", "301", "688", "689")):
        return 0.20
    if code.startswith(("8", "4")):
        return 0.30
    return 0.10


def add_ashare_indicators(prices: pd.DataFrame, concepts: pd.DataFrame | None = None) -> pd.DataFrame:
    frames = []
    concept_lookup = {}
    company_lookup = {}
    if concepts is not None and not concepts.empty:
        concept_lookup = concepts.set_index("ticker")["concept"].to_dict()
        if "company" in concepts.columns:
            company_lookup = concepts.set_index("ticker")["company"].to_dict()

    for ticker, group in prices.sort_values(["ticker", "date"]).groupby("ticker", sort=False):
        data = group.copy()
        close = data["close"]
        high = data["high"]
        low = data["low"]
        data["company"] = data.get("company", company_lookup.get(ticker, ""))
        data["concept"] = data.get("concept", concept_lookup.get(ticker, ""))
        for window in [5, 10, 20, 30, 60, 120, 250]:
            data[f"ma{window}"] = close.rolling(window).mean()
            data[f"momentum_{window}"] = close.pct_change(window)
        data["ema12"] = close.ewm(span=12, adjust=False).mean()
        data["ema26"] = close.ewm(span=26, adjust=False).mean()
        data["macd"] = data["ema12"] - data["ema26"]
        data["macd_signal"] = data["macd"].ewm(span=9, adjust=False).mean()
        data["macd_hist"] = data["macd"] - data["macd_signal"]
        data["rsi14"] = _rsi(close, 14)
        data["kdj_k"], data["kdj_d"], data["kdj_j"] = _kdj(high, low, close)
        data["bb_mid"] = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        data["bb_upper"] = data["bb_mid"] + 2 * bb_std
        data["bb_lower"] = data["bb_mid"] - 2 * bb_std
        data["atr14"] = _atr(high, low, close, 14)
        data["atr_pct"] = data["atr14"] / close
        data["return_1d"] = close.pct_change()
        data["volatility_20"] = data["return_1d"].rolling(20).std() * np.sqrt(252)
        data["volume_ratio"] = data["volume"] / data["volume"].rolling(20).mean()
        data["high_60d"] = high.rolling(60).max()
        data["low_60d"] = low.rolling(60).min()
        data["high_250d"] = high.rolling(250).max()
        data["low_250d"] = low.rolling(250).min()
        limit_pct = ashare_limit_pct(ticker, company_lookup.get(ticker, ""))
        data["limit_pct"] = limit_pct
        data["limit_up"] = data["return_1d"] >= (limit_pct - 0.005)
        data["limit_down"] = data["return_1d"] <= -(limit_pct - 0.005)
        data["ma_golden_cross"] = (data["ma5"] > data["ma20"]) & (data["ma5"].shift(1) <= data["ma20"].shift(1))
        data["ma_dead_cross"] = (data["ma5"] < data["ma20"]) & (data["ma5"].shift(1) >= data["ma20"].shift(1))
        data["macd_golden_cross"] = (data["macd"] > data["macd_signal"]) & (data["macd"].shift(1) <= data["macd_signal"].shift(1))
        data["macd_dead_cross"] = (data["macd"] < data["macd_signal"]) & (data["macd"].shift(1) >= data["macd_signal"].shift(1))
        data["ma20_breakout"] = (close > data["ma20"]) & (close.shift(1) <= data["ma20"].shift(1))
        data["ma20_breakdown"] = (close < data["ma20"]) & (close.shift(1) >= data["ma20"].shift(1))
        data["new_60d_high"] = close >= data["high_60d"].shift(1)
        data["new_60d_low"] = close <= data["low_60d"].shift(1)
        data["boll_breakout"] = close > data["bb_upper"]
        data["boll_breakdown"] = close < data["bb_lower"]
        data["volume_surge"] = data["volume_ratio"] >= 1.8
        frames.append(data)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def latest_snapshot(enriched: pd.DataFrame) -> pd.DataFrame:
    if enriched.empty:
        return pd.DataFrame()
    latest = enriched.sort_values(["ticker", "date"]).groupby("ticker", as_index=False).tail(1).copy()
    latest["trend_alignment"] = (
        (latest["close"] > latest["ma20"]).astype(int)
        + (latest["ma20"] > latest["ma60"]).astype(int)
        + (latest["ma60"] > latest["ma120"]).astype(int)
    )
    return latest.reset_index(drop=True)

