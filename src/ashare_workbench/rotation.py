import pandas as pd


def concept_rotation(enriched: pd.DataFrame, concepts: pd.DataFrame) -> pd.DataFrame:
    if enriched.empty or concepts.empty:
        return pd.DataFrame()
    latest = enriched.sort_values(["ticker", "date"]).groupby("ticker", as_index=False).tail(1)
    merged = latest.merge(concepts[["ticker", "concept"]], on="ticker", how="left", suffixes=("", "_map"))
    merged["concept"] = merged["concept"].where(merged["concept"].astype(str).str.len() > 0, merged["concept_map"])
    rows = []
    for concept, group in merged.groupby("concept"):
        rows.append(
            {
                "concept": concept,
                "members": group["ticker"].nunique(),
                "avg_1d": group["return_1d"].mean(),
                "avg_20d": group["momentum_20"].mean(),
                "avg_60d": group["momentum_60"].mean(),
                "avg_volume_ratio": group["volume_ratio"].mean(),
                "limit_ups": int(group["limit_up"].sum()),
                "leaders": ", ".join(group.sort_values("momentum_20", ascending=False)["ticker"].head(3).tolist()),
            }
        )
    return pd.DataFrame(rows).sort_values(["avg_20d", "avg_volume_ratio"], ascending=False).reset_index(drop=True)


def limit_up_ladder(enriched: pd.DataFrame, lookback: int = 8) -> pd.DataFrame:
    rows = []
    if enriched.empty:
        return pd.DataFrame()
    for ticker, group in enriched.sort_values(["ticker", "date"]).groupby("ticker", sort=False):
        recent = group.tail(lookback).copy()
        streak = 0
        for is_limit in reversed(recent["limit_up"].fillna(False).tolist()):
            if is_limit:
                streak += 1
            else:
                break
        latest = recent.tail(1).iloc[0]
        ladder = "首板" if streak == 1 else f"{streak}连板" if streak > 1 else "未涨停"
        rows.append(
            {
                "ticker": ticker,
                "company": latest.get("company", ""),
                "concept": latest.get("concept", ""),
                "date": latest["date"],
                "ladder": ladder,
                "limit_streak": streak,
                "close": latest["close"],
                "return_1d": latest["return_1d"],
                "volume_ratio": latest["volume_ratio"],
                "risk_note": "情绪过热，研究用" if streak >= 2 else "观察量价持续性" if streak == 1 else "无涨停情绪",
            }
        )
    return pd.DataFrame(rows).sort_values(["limit_streak", "volume_ratio"], ascending=False).reset_index(drop=True)

