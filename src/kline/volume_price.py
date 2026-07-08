from __future__ import annotations

import numpy as np
import pandas as pd


ACTION_COLOR = {
    "hold": "#2ed17c",
    "follow": "#d8cb5f",
    "wait": "#9ca39d",
    "reduce": "#ff5f64",
    "risk": "#ff78b7",
    "watch": "#a891ff",
}

SIGNAL_COLUMNS = ["date", "close", "volume_ratio", "price_zone", "volume_state", "price_state", "label", "action", "note"]


def analyze_volume_price_state(ohlcv: pd.DataFrame, benchmark: pd.DataFrame | None = None, lookback: int = 120) -> dict[str, pd.DataFrame | str]:
    data = _prepare_volume_price_frame(ohlcv)
    if data.empty:
        empty = pd.DataFrame(columns=SIGNAL_COLUMNS)
        return {"frame": data, "signals": empty, "markers": empty, "summary": "No volume-price state available."}

    recent = data.tail(max(30, int(lookback))).copy()
    signal_rows = []
    for _, row in recent.iterrows():
        signal_rows.extend(_daily_rule_hits(row))

    signals = pd.DataFrame(signal_rows)
    if signals.empty:
        signals = pd.DataFrame(columns=SIGNAL_COLUMNS)
    if not signals.empty:
        signals = signals.sort_values("date", ascending=False).reset_index(drop=True)

    latest = data.iloc[-1]
    latest_hits = _daily_rule_hits(latest)
    environment_note = _market_environment_note(benchmark)
    yearly_note = _yearly_line_note(latest)
    fall_note = _fall_speed_note(data)
    gap_note = _gap_note(data)
    headline = _headline(latest_hits, latest)
    summary_parts = [headline, yearly_note, gap_note, fall_note, environment_note]
    summary = " ".join(part for part in summary_parts if part)

    markers = _marker_frame(signals)
    return {
        "frame": data,
        "signals": signals,
        "markers": markers,
        "summary": summary,
    }


def latest_volume_price_note(ohlcv: pd.DataFrame, benchmark: pd.DataFrame | None = None) -> str:
    return str(analyze_volume_price_state(ohlcv, benchmark)["summary"])


def _prepare_volume_price_frame(ohlcv: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "open", "high", "low", "close", "volume"}
    if ohlcv.empty or not required.issubset(ohlcv.columns):
        return pd.DataFrame()

    data = ohlcv.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    for col in ["open", "high", "low", "close", "volume"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=["date", "open", "high", "low", "close", "volume"]).sort_values("date")
    close = data["close"]
    volume = data["volume"]
    data["return_1d"] = close.pct_change()
    data["return_5d"] = close.pct_change(5)
    data["volume_ma20"] = volume.rolling(20, min_periods=5).mean()
    data["volume_ratio"] = volume / data["volume_ma20"].replace(0, np.nan)
    data["high_120d"] = close.rolling(120, min_periods=20).max()
    data["low_120d"] = close.rolling(120, min_periods=20).min()
    data["range_position"] = (close - data["low_120d"]) / (data["high_120d"] - data["low_120d"]).replace(0, np.nan)
    data["ma20"] = close.rolling(20, min_periods=5).mean()
    data["ma60"] = close.rolling(60, min_periods=20).mean()
    data["ma200"] = close.rolling(200, min_periods=80).mean()
    data["ma200_slope_20"] = data["ma200"] / data["ma200"].shift(20) - 1
    data["price_state"] = np.select(
        [data["return_1d"] >= 0.012, data["return_1d"] <= -0.012],
        ["price_up", "price_down"],
        default="price_flat",
    )
    data["volume_state"] = np.select(
        [data["volume_ratio"] >= 1.8, data["volume_ratio"] <= 0.8],
        ["volume_surge", "volume_quiet"],
        default="volume_normal",
    )
    data["price_zone"] = np.select(
        [data["range_position"] >= 0.75, data["range_position"] <= 0.25],
        ["high_zone", "low_zone"],
        default="middle_zone",
    )
    return data.reset_index(drop=True)


def _daily_rule_hits(row: pd.Series) -> list[dict]:
    hits = []
    zone = row.get("price_zone")
    volume_state = row.get("volume_state")
    price_state = row.get("price_state")
    date = row.get("date")
    close = row.get("close")
    volume_ratio = row.get("volume_ratio")

    def add(label: str, action: str, note: str):
        hits.append(
            {
                "date": date,
                "close": close,
                "volume_ratio": volume_ratio,
                "price_zone": zone,
                "volume_state": volume_state,
                "price_state": price_state,
                "label": label,
                "action": action,
                "note": note,
            }
        )

    if zone == "high_zone" and volume_state == "volume_quiet":
        add("高位无量", "hold", "强势横盘更偏上涨中继，先拿着观察。")
    if zone == "high_zone" and volume_state == "volume_surge":
        add("高位放量", "reduce", "高位突然放量代表分歧变大，先撤或降仓。")
    if zone == "low_zone" and volume_state == "volume_quiet":
        add("低位无量", "wait", "资金还没启动，耐心等量。")
    if zone == "low_zone" and volume_state == "volume_surge" and price_state != "price_down":
        add("低位放量", "follow", "长期横盘后放量，进入启动观察。")
    if volume_state == "volume_surge" and price_state == "price_flat":
        add("量增价平", "risk", "量上来但价格推不动，注意变盘。")
    if volume_state == "volume_surge" and price_state == "price_up":
        add("量增价升", "follow", "量价齐升，多头占优。")
    if volume_state == "volume_normal" and price_state == "price_up":
        add("量平价升", "hold", "缩量或平量上推，说明筹码锁定较好。")
    if volume_state == "volume_normal" and price_state == "price_down":
        add("量平价跌", "reduce", "价格下行但没有承接，先离场或降仓。")
    return hits


def _marker_frame(signals: pd.DataFrame) -> pd.DataFrame:
    if signals.empty:
        return signals
    markers = signals.drop_duplicates(["date", "action"]).copy()
    markers["color"] = markers["action"].map(ACTION_COLOR).fillna("#a891ff")
    markers["marker_text"] = markers["label"]
    return markers[["date", "close", "label", "action", "note", "color", "marker_text"]]


def _headline(latest_hits: list[dict], latest: pd.Series) -> str:
    if latest_hits:
        primary = latest_hits[0]
        return f"当前量价状态：{primary['label']}，{primary['note']}"
    zone_map = {"high_zone": "高位", "low_zone": "低位", "middle_zone": "中位"}
    volume_map = {"volume_surge": "放量", "volume_quiet": "无量", "volume_normal": "量平"}
    price_map = {"price_up": "价升", "price_down": "价跌", "price_flat": "价平"}
    return (
        "当前量价状态："
        f"{zone_map.get(latest.get('price_zone'), '未知位置')}，"
        f"{volume_map.get(latest.get('volume_state'), '量能不明')}，"
        f"{price_map.get(latest.get('price_state'), '价格不明')}，暂时按观察处理。"
    )


def _yearly_line_note(latest: pd.Series) -> str:
    slope = latest.get("ma200_slope_20")
    close = latest.get("close")
    ma200 = latest.get("ma200")
    if pd.isna(slope) or pd.isna(ma200):
        return "年线数据不足，暂不判断大周期。"
    if slope > 0.015 and close >= ma200 * 0.98:
        return "年线向上且价格在年线附近或上方，回踩更偏机会。"
    if abs(slope) <= 0.01:
        return "年线走平，大周期开始进入危险观察区。"
    if slope < -0.01:
        return "年线下拐，大周期环境偏弱。"
    return "年线状态中性。"


def _gap_note(data: pd.DataFrame) -> str:
    if len(data) < 2:
        return ""
    prev = data.shift(1)
    gaps = data.loc[data["low"] > prev["high"]].copy()
    if gaps.empty:
        return "近期没有未补的向上缺口信号。"
    latest_gap = gaps.iloc[-1]
    later = data.loc[data["date"] > latest_gap["date"]]
    unfilled = later.empty or later["low"].min() > prev.loc[latest_gap.name, "high"]
    if unfilled:
        return "最近存在向上缺口且后续未完全回补，资金态度偏强。"
    return "最近向上缺口已回补，突破强度需要重新确认。"


def _fall_speed_note(data: pd.DataFrame) -> str:
    if len(data) < 10:
        return ""
    recent_5d = data["close"].iloc[-1] / data["close"].iloc[-6] - 1 if len(data) >= 6 else 0
    recent_20d = data["close"].iloc[-1] / data["close"].iloc[-21] - 1 if len(data) >= 21 else recent_5d
    if recent_5d <= -0.08:
        return "近期急跌，后续若有反弹也会更快，但先等止跌确认。"
    if recent_20d <= -0.08 and recent_5d > -0.03:
        return "近期更像阴跌，修复通常慢，别急着抄底。"
    return "下跌速度没有触发急跌/阴跌警报。"


def _market_environment_note(benchmark: pd.DataFrame | None) -> str:
    if benchmark is None or benchmark.empty:
        return ""
    env = _prepare_volume_price_frame(benchmark)
    if len(env) < 60:
        return ""
    latest = env.iloc[-1]
    if latest["close"] < latest["ma60"] and latest["ma20"] < latest["ma60"]:
        return "大盘环境破位，单票信号需要降权，优先空仓或轻仓。"
    return "大盘环境未触发破位空仓规则。"
