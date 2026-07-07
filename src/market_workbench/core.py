import numpy as np
import pandas as pd


STRATEGY_DESCRIPTIONS = {
    "trend_breakout": "60日新高 + 放量 + 多头排列，寻找趋势突破。",
    "ma_golden_cross": "MA5 上穿 MA20，配合价格站上 MA20。",
    "macd_golden": "MACD 金叉，动量开始修复。",
    "volume_price_surge": "放量上涨，观察资金突然进入。",
    "low_volatility_leader": "低波动、价格强于中期均线的稳健领涨候选。",
    "oversold_bounce": "RSI 超卖后靠近布林下轨，反弹观察。",
    "boll_breakout": "价格突破布林上轨，适合研究趋势扩张。",
    "bullish_alignment": "MA20 > MA60 > MA120，多周期趋势顺排。",
    "surge_ladder": "连续强势上涨或突破，观察市场情绪高度。",
    "pullback_to_support": "多头趋势中回踩 MA20/MA60 支撑。",
    "n_day_low_reversal": "接近 60 日低点后出现放量反弹。",
}


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
        [high - low, (high - previous_close).abs(), (low - previous_close).abs()],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(window).mean()


def surge_threshold(market: str) -> float:
    if market == "CRYPTO":
        return 0.045
    if market == "A_SHARE":
        return 0.095
    if market == "HK":
        return 0.04
    return 0.035


def add_market_indicators(prices: pd.DataFrame, metadata: pd.DataFrame | None = None) -> pd.DataFrame:
    if prices.empty:
        return pd.DataFrame()
    meta = pd.DataFrame() if metadata is None else metadata.copy()
    meta_lookup = meta.set_index("ticker").to_dict("index") if not meta.empty and "ticker" in meta.columns else {}
    frames = []
    for ticker, group in prices.sort_values(["ticker", "date"]).groupby("ticker", sort=False):
        data = group.copy()
        info = meta_lookup.get(str(ticker), {})
        for col, default in [
            ("market", info.get("market", "UNKNOWN")),
            ("asset_class", info.get("asset_class", "equity")),
            ("company", info.get("company", str(ticker))),
            ("concept", info.get("concept", "Unclassified")),
            ("currency", info.get("currency", "")),
        ]:
            if col not in data.columns:
                data[col] = default
            else:
                data[col] = data[col].fillna(default)
        close = data["close"]
        high = data["high"]
        low = data["low"]
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
        annualizer = 365 if str(data["market"].iloc[-1]) == "CRYPTO" else 252
        data["volatility_20"] = data["return_1d"].rolling(20).std() * np.sqrt(annualizer)
        data["volume_ratio"] = data["volume"] / data["volume"].rolling(20).mean()
        data["high_60d"] = high.rolling(60).max()
        data["low_60d"] = low.rolling(60).min()
        data["high_250d"] = high.rolling(250).max()
        data["low_250d"] = low.rolling(250).min()
        threshold = surge_threshold(str(data["market"].iloc[-1]))
        data["surge_threshold"] = threshold
        data["strong_surge"] = data["return_1d"] >= threshold
        data["limit_up"] = data["strong_surge"] & (data["market"] == "A_SHARE")
        data["limit_down"] = data["return_1d"] <= -threshold
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
        data["strong_surge"] = data["strong_surge"] | (data["new_60d_high"] & (data["volume_ratio"] >= 1.5))
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


def _score_row(row: pd.Series) -> tuple[int, list[str]]:
    score = 0
    reasons = []
    if row.get("trend_alignment", 0) >= 2:
        score += 14
        reasons.append("均线结构偏强")
    if row.get("momentum_20", 0) > 0:
        score += 10
        reasons.append("20日动量为正")
    if row.get("momentum_60", 0) > 0:
        score += 8
        reasons.append("60日动量为正")
    if row.get("volume_ratio", 0) >= 1.4:
        score += 8
        reasons.append("成交量放大")
    if row.get("rsi14", 50) < 72:
        score += 6
        reasons.append("未进入极端超买")
    if row.get("volatility_20", 1) < 0.8:
        score += 6
        reasons.append("波动受控")
    if row.get("new_60d_high", False):
        score += 8
        reasons.append("创60日新高")
    if row.get("strong_surge", False):
        score += 6
        reasons.append("强势突破/情绪激活")
    return min(score, 60), reasons


def screen_market(enriched: pd.DataFrame) -> pd.DataFrame:
    latest = latest_snapshot(enriched)
    if latest.empty:
        return pd.DataFrame()
    rows = []
    for _, row in latest.iterrows():
        score, base_reasons = _score_row(row)
        strategies = []
        if bool(row.get("new_60d_high", False)) and row.get("volume_ratio", 0) >= 1.3 and row.get("trend_alignment", 0) >= 2:
            strategies.append("trend_breakout")
        if bool(row.get("ma_golden_cross", False)) or (row.get("close", 0) > row.get("ma20", 0) and row.get("ma5", 0) > row.get("ma20", 0)):
            strategies.append("ma_golden_cross")
        if bool(row.get("macd_golden_cross", False)) or row.get("macd_hist", -1) > 0:
            strategies.append("macd_golden")
        if row.get("return_1d", 0) > min(row.get("surge_threshold", 0.035), 0.035) and row.get("volume_ratio", 0) >= 1.5:
            strategies.append("volume_price_surge")
        if row.get("volatility_20", 1) < 0.45 and row.get("close", 0) > row.get("ma60", 0):
            strategies.append("low_volatility_leader")
        if row.get("rsi14", 50) < 36 or row.get("close", 0) <= row.get("bb_lower", 0) * 1.02:
            strategies.append("oversold_bounce")
        if bool(row.get("boll_breakout", False)):
            strategies.append("boll_breakout")
        if row.get("close", 0) > row.get("ma20", 0) > row.get("ma60", 0) > row.get("ma120", 0):
            strategies.append("bullish_alignment")
        if bool(row.get("strong_surge", False)):
            strategies.append("surge_ladder")
        if row.get("ma60", 0) > 0 and row.get("close", 0) > row.get("ma60", 0) and abs(row.get("close", 0) / row.get("ma20", 1) - 1) < 0.035:
            strategies.append("pullback_to_support")
        if row.get("low_60d", 0) > 0 and row.get("close", 0) <= row.get("low_60d", 0) * 1.08 and row.get("return_1d", 0) > 0:
            strategies.append("n_day_low_reversal")
        if not strategies:
            strategies.append("watch_only")
        rows.append(
            {
                "market": row.get("market", ""),
                "asset_class": row.get("asset_class", ""),
                "ticker": row["ticker"],
                "company": row.get("company", ""),
                "concept": row.get("concept", ""),
                "date": row.get("date"),
                "close": row.get("close"),
                "score": score,
                "strategies": ", ".join(strategies),
                "reasons": " / ".join(base_reasons[:4]),
                "return_1d": row.get("return_1d"),
                "momentum_20": row.get("momentum_20"),
                "momentum_60": row.get("momentum_60"),
                "volume_ratio": row.get("volume_ratio"),
                "rsi14": row.get("rsi14"),
                "volatility_20": row.get("volatility_20"),
                "strong_surge": bool(row.get("strong_surge", False)),
                "limit_up": bool(row.get("limit_up", False)),
            }
        )
    return pd.DataFrame(rows).sort_values(["score", "momentum_20"], ascending=False).reset_index(drop=True)


def filter_screener(screened: pd.DataFrame, market: str = "ALL", strategy: str = "ALL", min_score: int = 0) -> pd.DataFrame:
    if screened.empty:
        return screened
    view = screened.loc[screened["score"] >= min_score].copy()
    if market != "ALL":
        view = view.loc[view["market"] == market]
    if strategy != "ALL":
        view = view.loc[view["strategies"].str.contains(strategy, regex=False, na=False)]
    return view.reset_index(drop=True)


def market_rotation(enriched: pd.DataFrame) -> pd.DataFrame:
    if enriched.empty:
        return pd.DataFrame()
    latest = latest_snapshot(enriched)
    rows = []
    for (market, concept), group in latest.groupby(["market", "concept"]):
        rows.append(
            {
                "market": market,
                "concept": concept,
                "members": group["ticker"].nunique(),
                "avg_1d": group["return_1d"].mean(),
                "avg_20d": group["momentum_20"].mean(),
                "avg_60d": group["momentum_60"].mean(),
                "avg_volume_ratio": group["volume_ratio"].mean(),
                "surge_count": int(group["strong_surge"].sum()),
                "leaders": ", ".join(group.sort_values("momentum_20", ascending=False)["ticker"].astype(str).head(3).tolist()),
            }
        )
    return pd.DataFrame(rows).sort_values(["avg_20d", "avg_volume_ratio"], ascending=False).reset_index(drop=True)


def surge_ladder(enriched: pd.DataFrame, lookback: int = 8) -> pd.DataFrame:
    rows = []
    if enriched.empty:
        return pd.DataFrame()
    for ticker, group in enriched.sort_values(["ticker", "date"]).groupby("ticker", sort=False):
        recent = group.tail(lookback).copy()
        streak = 0
        for is_surge in reversed(recent["strong_surge"].fillna(False).tolist()):
            if is_surge:
                streak += 1
            else:
                break
        latest = recent.tail(1).iloc[0]
        if latest.get("market", "") == "A_SHARE":
            ladder = "首板" if streak == 1 else f"{streak}连板" if streak > 1 else "未涨停"
        else:
            ladder = "强势首日" if streak == 1 else f"{streak}日强势" if streak > 1 else "未触发"
        rows.append(
            {
                "market": latest.get("market", ""),
                "ticker": ticker,
                "company": latest.get("company", ""),
                "concept": latest.get("concept", ""),
                "date": latest["date"],
                "ladder": ladder,
                "surge_streak": streak,
                "close": latest["close"],
                "return_1d": latest["return_1d"],
                "volume_ratio": latest["volume_ratio"],
                "risk_note": "情绪过热，研究用" if streak >= 2 else "观察量价持续性" if streak == 1 else "无强势情绪",
            }
        )
    return pd.DataFrame(rows).sort_values(["surge_streak", "volume_ratio"], ascending=False).reset_index(drop=True)


def monitor_verdict(latest_row: pd.Series) -> dict:
    alerts = []
    market = latest_row.get("market", "")
    if latest_row.get("strong_surge", False):
        alerts.append("强势突破/情绪激活")
    if market == "A_SHARE" and latest_row.get("limit_up", False):
        alerts.append("涨停情绪激活")
    if latest_row.get("volume_ratio", 0) >= 2:
        alerts.append("成交量异常放大")
    if latest_row.get("rsi14", 50) >= 78:
        alerts.append("RSI过热")
    if latest_row.get("close", 0) < latest_row.get("ma60", 0):
        alerts.append("跌破MA60")
    if latest_row.get("volatility_20", 0) >= (1.25 if market == "CRYPTO" else 0.8):
        alerts.append("年化波动过高")
    if not alerts:
        alerts.append("无硬性警报")
    status = "review" if len(alerts) >= 2 else "watch"
    if "跌破MA60" in alerts or "年化波动过高" in alerts:
        status = "risk"
    return {"status": status, "alerts": " / ".join(alerts)}
