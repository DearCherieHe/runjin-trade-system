from __future__ import annotations

import numpy as np
import pandas as pd


def launch_point_analysis(ohlcv: pd.DataFrame, lookback: int = 180) -> dict[str, pd.DataFrame | str]:
    data = _prepare_launch_frame(ohlcv).tail(max(lookback, 80)).copy()
    if data.empty:
        return {
            "summary": "起涨点研究：数据不足，无法判断。",
            "signals": pd.DataFrame(columns=_signal_columns()),
            "levels": pd.DataFrame(columns=["level_type", "price", "distance_pct", "evidence", "note"]),
            "plan": pd.DataFrame(columns=["rule", "value", "note"]),
        }

    levels = _control_levels(data)
    signals = _launch_signals(data, levels)
    plan = _risk_plan(data, levels, signals)
    summary = _summary(data, signals, levels)
    return {"summary": summary, "signals": signals, "levels": levels, "plan": plan}


def _prepare_launch_frame(ohlcv: pd.DataFrame) -> pd.DataFrame:
    if ohlcv is None or ohlcv.empty:
        return pd.DataFrame()
    data = ohlcv.copy()
    if "date" not in data.columns:
        date_col = "datetime" if "datetime" in data.columns else data.columns[0]
        data = data.rename(columns={date_col: "date"})
    required = {"date", "open", "high", "low", "close", "volume"}
    if not required.issubset(data.columns):
        return pd.DataFrame()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    for col in ["open", "high", "low", "close", "volume"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=["date", "open", "high", "low", "close"]).sort_values("date").reset_index(drop=True)
    close = data["close"]
    volume = data["volume"].replace(0, np.nan)
    data["ma20"] = data.get("ma20", close.rolling(20).mean())
    data["ma60"] = data.get("ma60", close.rolling(60).mean())
    data["bb_mid"] = data.get("bb_mid", close.rolling(20).mean())
    std = close.rolling(20).std()
    data["bb_upper"] = data.get("bb_upper", data["bb_mid"] + 2 * std)
    data["bb_lower"] = data.get("bb_lower", data["bb_mid"] - 2 * std)
    data["bb_width"] = (data["bb_upper"] - data["bb_lower"]) / data["bb_mid"].replace(0, np.nan)
    data["volume_ratio"] = volume / volume.rolling(20).mean()
    data["body_pct"] = (data["close"] - data["open"]) / data["open"].replace(0, np.nan)
    data["range_pct"] = (data["high"] - data["low"]) / data["close"].replace(0, np.nan)
    data["prior_20_high"] = data["high"].shift(1).rolling(20).max()
    data["prior_20_low"] = data["low"].shift(1).rolling(20).min()
    data["prev_high"] = data["high"].shift(1)
    data["prev_low"] = data["low"].shift(1)
    data["atr14"] = _atr(data, 14)
    return data


def _launch_signals(data: pd.DataFrame, levels: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for idx, row in data.tail(80).iterrows():
        if idx < 30:
            continue
        rows.extend(_pattern_launch(row, data.iloc[: idx + 1]))
        rows.extend(_gap_launch(row, data.iloc[: idx + 1]))
        rows.extend(_bollinger_launch(row, data.iloc[: idx + 1]))
        rows.extend(_control_launch(row, levels))
    if not rows:
        return pd.DataFrame(columns=_signal_columns())
    signals = pd.DataFrame(rows)
    signals = signals.sort_values(["date", "confidence"], ascending=[False, False]).reset_index(drop=True)
    return signals[_signal_columns()]


def _pattern_launch(row: pd.Series, history: pd.DataFrame) -> list[dict]:
    close = row["close"]
    bullish_body = row["body_pct"] > 0.025
    breakout = close > row.get("prior_20_high", np.nan)
    volume_confirm = row.get("volume_ratio", np.nan) >= 1.25
    ma_confirm = close > row.get("ma20", np.nan) and row.get("ma20", np.nan) >= row.get("ma60", np.nan) * 0.98
    if not (bullish_body and breakout):
        return []
    confidence = 55 + 15 * bool(volume_confirm) + 10 * bool(ma_confirm)
    stop = min(row["low"], history["low"].tail(10).min())
    return [_signal(row, "pattern_launch", "review" if confidence < 75 else "watch", confidence, row["low"], close, stop, "Large bullish breakout candle. Higher-quality if volume and MA trend confirm.")]


def _gap_launch(row: pd.Series, history: pd.DataFrame) -> list[dict]:
    if pd.isna(row.get("prev_high")) or row["open"] <= row["prev_high"] * 1.01:
        return []
    holds_gap = row["close"] >= row["open"] and row["low"] >= row["prev_high"] * 0.995
    volume_confirm = row.get("volume_ratio", np.nan) >= 1.2
    confidence = 60 + 15 * bool(holds_gap) + 10 * bool(volume_confirm)
    stop = min(row["prev_high"], row["low"])
    return [_signal(row, "gap_launch", "watch" if holds_gap else "review", confidence, row["prev_high"], row["close"], stop, "Gap launch candidate. Failed gap fill is the key invalidation signal.")]


def _bollinger_launch(row: pd.Series, history: pd.DataFrame) -> list[dict]:
    if len(history) < 40 or pd.isna(row.get("bb_mid")) or pd.isna(row.get("bb_width")):
        return []
    prev = history.iloc[-2]
    squeeze_threshold = history["bb_width"].rolling(80, min_periods=30).quantile(0.30).iloc[-1]
    squeeze = row["bb_width"] <= squeeze_threshold * 1.15 if pd.notna(squeeze_threshold) else False
    lower_reclaim = prev["close"] <= prev["bb_lower"] * 1.02 and row["close"] > row["bb_mid"]
    mid_break = prev["close"] <= prev["bb_mid"] and row["close"] > row["bb_mid"] and row["volume_ratio"] >= 1.1
    upper_expansion = row["close"] > row["bb_upper"] and row["volume_ratio"] >= 1.2
    if not (lower_reclaim or (squeeze and mid_break) or upper_expansion):
        return []
    confidence = 58 + 12 * bool(squeeze) + 12 * bool(row["volume_ratio"] >= 1.2) + 8 * bool(row["close"] > row["ma20"])
    stop = min(row["bb_lower"], history["low"].tail(8).min())
    return [_signal(row, "bollinger_launch", "watch" if confidence >= 72 else "review", confidence, row["bb_mid"], row["close"], stop, "Bollinger launch candidate from reclaim, squeeze release, or upper-band expansion.")]


def _control_launch(row: pd.Series, levels: pd.DataFrame) -> list[dict]:
    if levels.empty:
        return []
    support = levels.loc[levels["level_type"] == "support"]
    if support.empty:
        return []
    nearest = support.iloc[0]
    near_support = abs(float(nearest["distance_pct"])) <= 4
    rebound = row["close"] > row["open"] and row["close"] > row.get("ma20", np.nan)
    if not (near_support and rebound):
        return []
    confidence = 62 + 10 * bool(row.get("volume_ratio", np.nan) >= 1.15) + 8 * bool(row["close"] > row.get("ma60", np.nan))
    stop = min(float(nearest["price"]) * 0.985, row["low"])
    return [_signal(row, "control_point_launch", "review" if confidence < 75 else "watch", confidence, float(nearest["price"]), row["close"], stop, "Approximate original-control-point rebound near high-evidence support. Research-only proxy.")]


def _control_levels(data: pd.DataFrame) -> pd.DataFrame:
    latest_close = float(data["close"].iloc[-1])
    pivots = []
    for idx in range(3, len(data) - 3):
        window = data.iloc[idx - 3 : idx + 4]
        row = data.iloc[idx]
        if row["low"] <= window["low"].min():
            pivots.append(("support", row["low"], row["volume_ratio"], row["date"]))
        if row["high"] >= window["high"].max():
            pivots.append(("resistance", row["high"], row["volume_ratio"], row["date"]))
    if not pivots:
        return pd.DataFrame(columns=["level_type", "price", "distance_pct", "evidence", "note"])
    rows = []
    for level_type, price, volume_ratio, date in pivots[-80:]:
        if price <= 0:
            continue
        distance = (latest_close / price - 1) * 100
        if level_type == "support" and price > latest_close:
            continue
        if level_type == "resistance" and price < latest_close:
            continue
        evidence = min(100, 45 + abs(distance) * -1.5 + min(float(volume_ratio) if pd.notna(volume_ratio) else 1, 3) * 15)
        rows.append(
            {
                "level_type": level_type,
                "price": float(price),
                "distance_pct": float(distance),
                "evidence": float(max(0, evidence)),
                "note": f"Pivot from {pd.to_datetime(date).strftime('%Y-%m-%d')}; proxy for control support/resistance.",
            }
        )
    if not rows:
        return pd.DataFrame(columns=["level_type", "price", "distance_pct", "evidence", "note"])
    levels = pd.DataFrame(rows).sort_values(["level_type", "evidence"], ascending=[True, False])
    return levels.groupby("level_type", as_index=False).head(3).reset_index(drop=True)


def _risk_plan(data: pd.DataFrame, levels: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    latest = data.iloc[-1]
    atr = latest.get("atr14", np.nan)
    atr_stop = latest["close"] - 2 * atr if pd.notna(atr) else np.nan
    control_support = levels.loc[levels["level_type"] == "support", "price"].max() if not levels.empty and (levels["level_type"] == "support").any() else np.nan
    hard_stop = np.nanmin([value for value in [atr_stop, control_support * 0.985 if pd.notna(control_support) else np.nan, latest["low"]] if pd.notna(value)])
    rows = [
        {"rule": "hard_stop_required", "value": f"{hard_stop:.2f}" if pd.notna(hard_stop) else "N/A", "note": "Every launch-point study must define invalidation before entry."},
        {"rule": "trailing_stop_reference", "value": f"{(latest['close'] - 2.5 * atr):.2f}" if pd.notna(atr) else "N/A", "note": "If price moves favorably, prefer passive exit by trailing stop instead of guessing the top."},
        {"rule": "avoid_noise_rule", "value": "major launch only", "note": "This panel ignores short random wiggles; it looks for larger structure plus timing confirmation."},
        {"rule": "latest_candidate_count", "value": int((signals["date"] == signals["date"].max()).sum()) if not signals.empty else 0, "note": "Multiple independent launch types on the same bar deserve more attention, not automatic execution."},
    ]
    return pd.DataFrame(rows)


def _signal(row: pd.Series, signal_type: str, status: str, confidence: float, entry_low: float, entry_high: float, stop: float, note: str) -> dict:
    close = float(row["close"])
    trail = close - 2.5 * float(row["atr14"]) if pd.notna(row.get("atr14")) else np.nan
    return {
        "date": row["date"],
        "signal_type": signal_type,
        "status": status,
        "confidence": round(float(min(max(confidence, 0), 100)), 1),
        "entry_zone": f"{float(entry_low):.2f} - {float(entry_high):.2f}",
        "stop_level": round(float(stop), 2) if pd.notna(stop) else np.nan,
        "trailing_stop": round(float(trail), 2) if pd.notna(trail) else np.nan,
        "note": note,
    }


def _summary(data: pd.DataFrame, signals: pd.DataFrame, levels: pd.DataFrame) -> str:
    latest_close = data["close"].iloc[-1]
    if signals.empty:
        return f"起涨点研究：当前没有高质量起涨点共振。最新价 {latest_close:.2f}，先等形态、跳空、布林带或控制点出现更清晰证据。"
    latest_date = signals["date"].max()
    latest = signals.loc[signals["date"] == latest_date]
    best = latest.sort_values("confidence", ascending=False).iloc[0]
    return f"起涨点研究：最近候选为 {best['signal_type']}，置信分 {best['confidence']}，状态 {best['status']}。先看止损是否清晰，再看是否有多信号共振。"


def _atr(data: pd.DataFrame, window: int = 14) -> pd.Series:
    prev_close = data["close"].shift(1)
    tr = pd.concat(
        [
            data["high"] - data["low"],
            (data["high"] - prev_close).abs(),
            (data["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window).mean()


def _signal_columns() -> list[str]:
    return ["date", "signal_type", "status", "confidence", "entry_zone", "stop_level", "trailing_stop", "note"]
