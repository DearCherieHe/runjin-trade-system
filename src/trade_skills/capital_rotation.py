import pandas as pd


DEFAULT_COHORTS = {
    "Semiconductors": ["NVDA", "AMD", "AVGO", "TSM", "ARM"],
    "AI Software": ["PLTR", "SOUN"],
    "EV Autonomy": ["TSLA"],
    "AI Cloud": ["CRWV"],
    "Storage": ["SNDK"],
}


def cohort_rotation(prices: pd.DataFrame, cohorts: dict[str, list[str]] | None = None, lookback: int = 63) -> dict:
    cohorts = cohorts or DEFAULT_COHORTS
    if prices.empty:
        return {"scores": pd.DataFrame(), "curves": pd.DataFrame(), "label": "No data"}
    data = prices.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data.dropna(subset=["date"]).sort_values(["ticker", "date"])
    curves = []
    rows = []
    for cohort, tickers in cohorts.items():
        members = data.loc[data["ticker"].isin(tickers)].copy()
        if members.empty:
            continue
        wide = members.pivot_table(index="date", columns="ticker", values="close").sort_index().ffill()
        normalized = wide / wide.iloc[0]
        curve = normalized.mean(axis=1).rename(cohort).reset_index()
        curves.append(curve)
        tail = curve.tail(lookback + 1)
        ret = tail[cohort].iloc[-1] / tail[cohort].iloc[0] - 1 if len(tail) > 1 else 0
        breadth = (wide.tail(lookback + 1).iloc[-1] / wide.tail(lookback + 1).iloc[0] - 1 > 0).mean() if len(wide) > lookback else 0
        rows.append({"cohort": cohort, "members": ", ".join([t for t in tickers if t in wide.columns]), "lookback_return": ret, "positive_breadth": breadth})
    if not curves:
        return {"scores": pd.DataFrame(), "curves": pd.DataFrame(), "label": "No cohort overlap"}
    merged = curves[0]
    for curve in curves[1:]:
        merged = merged.merge(curve, on="date", how="outer")
    scores = pd.DataFrame(rows).sort_values("lookback_return", ascending=False).reset_index(drop=True)
    leader = scores.iloc[0]["cohort"] if not scores.empty else "Unknown"
    laggard = scores.iloc[-1]["cohort"] if len(scores) > 1 else "Unknown"
    if leader == "Semiconductors":
        label = "AI hardware leadership"
    elif leader == "AI Software":
        label = "capital rotating into software narratives"
    elif leader == "Storage":
        label = "storage cycle bid"
    else:
        label = f"{leader} leadership over {laggard}"
    return {"scores": scores, "curves": merged.sort_values("date"), "label": label}

