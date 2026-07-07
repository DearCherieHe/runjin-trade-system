import pandas as pd

from .indicators import latest_snapshot


STRATEGY_DESCRIPTIONS = {
    "trend_breakout": "60日新高 + 放量 + 多头排列，寻找趋势突破。",
    "ma_golden_cross": "MA5 上穿 MA20，配合价格站上 MA20。",
    "macd_golden": "MACD 金叉，动量开始修复。",
    "volume_price_surge": "放量上涨，观察资金突然进入。",
    "low_volatility_leader": "低波动、价格强于中期均线的稳健领涨候选。",
    "oversold_bounce": "RSI 超卖后靠近布林下轨，反弹观察。",
    "boll_breakout": "价格突破布林上轨，适合研究趋势扩张。",
    "bullish_alignment": "MA20 > MA60 > MA120，多周期趋势顺排。",
    "consecutive_limit_ups": "连续涨停梯队，适合观察情绪高度。",
    "pullback_to_support": "多头趋势中回踩 MA20/MA60 支撑。",
    "n_day_low_reversal": "接近 60 日低点后出现放量反弹。",
}


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
    if row.get("volatility_20", 1) < 0.65:
        score += 6
        reasons.append("波动受控")
    if row.get("new_60d_high", False):
        score += 8
        reasons.append("创60日新高")
    if row.get("limit_up", False):
        score += 6
        reasons.append("涨停情绪")
    return min(score, 60), reasons


def screen_ashare(enriched: pd.DataFrame) -> pd.DataFrame:
    latest = latest_snapshot(enriched)
    if latest.empty:
        return pd.DataFrame()
    rows = []
    for _, row in latest.iterrows():
        ticker = row["ticker"]
        score, base_reasons = _score_row(row)
        strategies = []
        if bool(row.get("new_60d_high", False)) and row.get("volume_ratio", 0) >= 1.3 and row.get("trend_alignment", 0) >= 2:
            strategies.append("trend_breakout")
        if bool(row.get("ma_golden_cross", False)) or (row.get("close", 0) > row.get("ma20", 0) and row.get("ma5", 0) > row.get("ma20", 0)):
            strategies.append("ma_golden_cross")
        if bool(row.get("macd_golden_cross", False)) or row.get("macd_hist", -1) > 0:
            strategies.append("macd_golden")
        if row.get("return_1d", 0) > 0.025 and row.get("volume_ratio", 0) >= 1.5:
            strategies.append("volume_price_surge")
        if row.get("volatility_20", 1) < 0.42 and row.get("close", 0) > row.get("ma60", 0):
            strategies.append("low_volatility_leader")
        if row.get("rsi14", 50) < 36 or row.get("close", 0) <= row.get("bb_lower", 0) * 1.02:
            strategies.append("oversold_bounce")
        if bool(row.get("boll_breakout", False)):
            strategies.append("boll_breakout")
        if row.get("close", 0) > row.get("ma20", 0) > row.get("ma60", 0) > row.get("ma120", 0):
            strategies.append("bullish_alignment")
        if bool(row.get("limit_up", False)):
            strategies.append("consecutive_limit_ups")
        if row.get("ma60", 0) > 0 and row.get("close", 0) > row.get("ma60", 0) and abs(row.get("close", 0) / row.get("ma20", 1) - 1) < 0.035:
            strategies.append("pullback_to_support")
        if row.get("low_60d", 0) > 0 and row.get("close", 0) <= row.get("low_60d", 0) * 1.08 and row.get("return_1d", 0) > 0:
            strategies.append("n_day_low_reversal")
        if not strategies:
            strategies.append("watch_only")
        rows.append(
            {
                "ticker": ticker,
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
                "limit_up": bool(row.get("limit_up", False)),
            }
        )
    return pd.DataFrame(rows).sort_values(["score", "momentum_20"], ascending=False).reset_index(drop=True)


def filter_screener(screened: pd.DataFrame, strategy: str = "ALL", min_score: int = 0) -> pd.DataFrame:
    if screened.empty:
        return screened
    view = screened.loc[screened["score"] >= min_score].copy()
    if strategy != "ALL":
        view = view.loc[view["strategies"].str.contains(strategy, regex=False, na=False)]
    return view.reset_index(drop=True)

