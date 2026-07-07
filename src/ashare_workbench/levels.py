import numpy as np
import pandas as pd


def key_price_levels(symbol_df: pd.DataFrame, lookback: int = 120) -> pd.DataFrame:
    if symbol_df.empty:
        return pd.DataFrame()
    data = symbol_df.sort_values("date").tail(lookback).copy()
    latest = data.tail(1).iloc[0]
    high = float(data["high"].max())
    low = float(data["low"].min())
    close = float(latest["close"])
    pivot = (float(latest["high"]) + float(latest["low"]) + close) / 3
    atr = float(data["atr14"].dropna().iloc[-1]) if "atr14" in data and data["atr14"].notna().any() else close * 0.03
    volume_profile = data.copy()
    bins = np.linspace(float(data["low"].min()), float(data["high"].max()), 18)
    volume_profile["price_bin"] = pd.cut(volume_profile["close"], bins=bins, include_lowest=True)
    poc_row = volume_profile.groupby("price_bin", observed=True)["volume"].sum().sort_values(ascending=False).head(1)
    poc = float(poc_row.index[0].mid) if not poc_row.empty else close
    fib_382 = high - (high - low) * 0.382
    fib_618 = high - (high - low) * 0.618
    previous = data.shift(1)
    gap_mask = (data["low"] > previous["high"]) | (data["high"] < previous["low"])
    latest_gap = data.loc[gap_mask].tail(1)
    gap_level = None
    if not latest_gap.empty:
        row = latest_gap.iloc[0]
        prev = data.loc[data["date"] < row["date"]].tail(1)
        if not prev.empty:
            gap_level = (float(row["low"]) + float(prev["high"].iloc[0])) / 2 if row["low"] > prev["high"].iloc[0] else (float(row["high"]) + float(prev["low"].iloc[0])) / 2
    rows = [
        ("Last close", close, "当前价格", "context"),
        ("Volume POC", poc, "近似筹码成交密集区", "support/resistance"),
        ("Pivot", pivot, "上一根K线枢轴", "support/resistance"),
        ("R1", 2 * pivot - float(latest["low"]), "枢轴压力", "resistance"),
        ("S1", 2 * pivot - float(latest["high"]), "枢轴支撑", "support"),
        ("60D high", float(data["high"].tail(60).max()), "60日高点", "resistance"),
        ("60D low", float(data["low"].tail(60).min()), "60日低点", "support"),
        ("ATR stop 1x", close - atr, "1倍ATR防守位", "risk"),
        ("ATR stop 2x", close - 2 * atr, "2倍ATR防守位", "risk"),
        ("Fib 38.2", fib_382, "波段回撤位", "support/resistance"),
        ("Fib 61.8", fib_618, "深回撤观察位", "support/resistance"),
    ]
    if gap_level is not None:
        rows.append(("Latest gap midpoint", gap_level, "最近缺口中轴", "support/resistance"))
    out = pd.DataFrame(rows, columns=["level", "price", "meaning", "type"])
    out["distance_to_close"] = out["price"] / close - 1
    return out


def monitor_verdict(latest_row: pd.Series) -> dict:
    alerts = []
    if latest_row.get("limit_up", False):
        alerts.append("涨停情绪激活")
    if latest_row.get("volume_ratio", 0) >= 2:
        alerts.append("成交量异常放大")
    if latest_row.get("rsi14", 50) >= 78:
        alerts.append("RSI过热")
    if latest_row.get("close", 0) < latest_row.get("ma60", 0):
        alerts.append("跌破MA60")
    if latest_row.get("volatility_20", 0) >= 0.8:
        alerts.append("年化波动过高")
    if not alerts:
        alerts.append("无硬性警报")
    status = "review" if len(alerts) >= 2 else "watch"
    if "跌破MA60" in alerts or "年化波动过高" in alerts:
        status = "risk"
    return {"status": status, "alerts": " / ".join(alerts)}

