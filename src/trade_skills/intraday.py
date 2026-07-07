import numpy as np
import pandas as pd


def _time_col(df: pd.DataFrame) -> str:
    if "datetime" in df.columns:
        return "datetime"
    return "date"


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = -delta.clip(upper=0).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    col = _time_col(df)
    data = df.copy()
    data[col] = pd.to_datetime(data[col], errors="coerce")
    data = data.dropna(subset=[col]).sort_values(col).set_index(col)
    if len(data) < 2:
        return data.reset_index().rename(columns={col: "datetime"})
    out = data.resample(rule).agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna().reset_index()
    return out.rename(columns={col: "datetime"})


def _tf_signal(df: pd.DataFrame, label: str) -> dict:
    data = df.copy().sort_values(_time_col(df))
    close = data["close"]
    ema9 = close.ewm(span=9, adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    ema55 = close.ewm(span=55, adjust=False).mean()
    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    signal = macd.ewm(span=9, adjust=False).mean()
    rsi14 = _rsi(close)
    latest = data.iloc[-1]
    bias_score = 0
    bias_score += 1 if ema9.iloc[-1] > ema21.iloc[-1] > ema55.iloc[-1] else -1 if ema9.iloc[-1] < ema21.iloc[-1] < ema55.iloc[-1] else 0
    bias_score += 1 if macd.iloc[-1] > signal.iloc[-1] else -1
    bias_score += 1 if latest["close"] > ema21.iloc[-1] else -1
    direction = "long" if bias_score >= 2 else "short" if bias_score <= -2 else "neutral"
    atr = pd.concat(
        [
            data["high"] - data["low"],
            (data["high"] - close.shift()).abs(),
            (data["low"] - close.shift()).abs(),
        ],
        axis=1,
    ).max(axis=1).rolling(14).mean().iloc[-1]
    atr = float(atr) if pd.notna(atr) and atr > 0 else float(latest["close"] * 0.02)
    entry = float(latest["close"])
    stop = entry - atr if direction == "long" else entry + atr if direction == "short" else entry
    target = entry + 2 * atr if direction == "long" else entry - 2 * atr if direction == "short" else entry
    return {
        "timeframe": label,
        "bars": len(data),
        "direction": direction,
        "bias_score": bias_score,
        "anchor_price": round(entry, 4),
        "rsi14": round(float(rsi14.iloc[-1]), 1) if pd.notna(rsi14.iloc[-1]) else None,
        "macd_state": "bullish" if macd.iloc[-1] > signal.iloc[-1] else "bearish",
        "entry": round(entry, 4),
        "stop": round(stop, 4),
        "target": round(target, 4),
        "note": "Research-only scenario. No order is generated.",
    }


def intraday_signal_pack(raw: pd.DataFrame) -> dict:
    if raw.empty:
        return {"signals": pd.DataFrame(), "scenarios": pd.DataFrame(), "risk_gate": pd.DataFrame()}
    data = raw.copy()
    col = _time_col(data)
    data[col] = pd.to_datetime(data[col], errors="coerce")
    data = data.dropna(subset=[col]).sort_values(col)
    if len(data) < 60:
        return {"signals": pd.DataFrame(), "scenarios": pd.DataFrame(), "risk_gate": pd.DataFrame([{"gate": "history", "status": "block", "reason": "Need at least 60 bars"}])}

    one_hour = _resample(data, "1h")
    fifteen = _resample(data, "15min") if data[col].diff().median() <= pd.Timedelta(minutes=15) else one_hour.copy()
    five = _resample(data, "5min") if data[col].diff().median() <= pd.Timedelta(minutes=5) else one_hour.copy()
    signals = pd.DataFrame([_tf_signal(five, "5m proxy"), _tf_signal(fifteen, "15m proxy"), _tf_signal(one_hour, "1h")])
    long_votes = int((signals["direction"] == "long").sum())
    short_votes = int((signals["direction"] == "short").sum())
    if long_votes > short_votes:
        base = "long"
        probability = 45 + 10 * long_votes
    elif short_votes > long_votes:
        base = "short"
        probability = 45 + 10 * short_votes
    else:
        base = "neutral"
        probability = 40
    scenarios = pd.DataFrame(
        [
            {"scenario": "primary", "direction": base, "probability": min(probability, 70), "action": "observe setup, require confirmation"},
            {"scenario": "range", "direction": "neutral", "probability": 25, "action": "reduce size or wait"},
            {"scenario": "failure", "direction": "opposite", "probability": max(10, 100 - min(probability, 70) - 25), "action": "respect stop/invalidation"},
        ]
    )
    latest_close = float(data["close"].iloc[-1])
    vol = data["close"].pct_change().rolling(24).std().iloc[-1]
    risk_gate = pd.DataFrame(
        [
            {"gate": "leverage", "status": "pass", "reason": "RunJin V0.1 is paper/research only"},
            {"gate": "event", "status": "review", "reason": "Check earnings/FOMC/news before acting"},
            {"gate": "volatility", "status": "review" if pd.notna(vol) and vol > 0.05 else "pass", "reason": f"Latest close {latest_close:.2f}; rolling volatility {0 if pd.isna(vol) else vol:.2%}"},
        ]
    )
    return {"signals": signals, "scenarios": scenarios, "risk_gate": risk_gate}

