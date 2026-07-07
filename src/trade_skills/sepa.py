import numpy as np
import pandas as pd


def _sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).mean()


def _relative_excess(df: pd.DataFrame, benchmark: pd.DataFrame, lookback: int) -> float | None:
    if benchmark.empty or len(df) <= lookback:
        return None
    left = df[["date", "close"]].rename(columns={"close": "asset_close"})
    right = benchmark[["date", "close"]].rename(columns={"close": "bench_close"})
    merged = left.merge(right, on="date", how="inner").sort_values("date")
    if len(merged) <= lookback:
        return None
    asset_return = merged["asset_close"].iloc[-1] / merged["asset_close"].iloc[-lookback - 1] - 1
    bench_return = merged["bench_close"].iloc[-1] / merged["bench_close"].iloc[-lookback - 1] - 1
    return float((asset_return - bench_return) * 100)


def sepa_dashboard(ticker_df: pd.DataFrame, benchmark_df: pd.DataFrame | None = None) -> dict:
    data = ticker_df.copy()
    if data.empty:
        return {"summary": pd.DataFrame(), "checks": pd.DataFrame(), "levels": pd.DataFrame(), "verdict": "NO DATA"}
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    data["ma50"] = _sma(data["close"], 50)
    data["ma150"] = _sma(data["close"], 150)
    data["ma200"] = _sma(data["close"], 200)
    data["vol20"] = _sma(data["volume"], 20) if "volume" in data.columns else np.nan
    if len(data) < 200:
        return {
            "summary": pd.DataFrame([{"metric": "bars", "value": len(data)}, {"metric": "status", "value": "Need 200+ bars"}]),
            "checks": pd.DataFrame(),
            "levels": data,
            "verdict": "INSUFFICIENT HISTORY",
        }

    row = data.iloc[-1]
    last = float(row["close"])
    ma50 = float(row["ma50"])
    ma150 = float(row["ma150"])
    ma200 = float(row["ma200"])
    ma200_1m = data["ma200"].iloc[-22] if len(data) >= 222 else np.nan
    window = data.tail(min(252, len(data)))
    high52w = float(window["high"].max())
    low52w = float(window["low"].min())
    benchmark = benchmark_df if benchmark_df is not None else pd.DataFrame()
    rs126 = _relative_excess(data, benchmark, 126)

    checks = [
        ("Price above 150MA and 200MA", last > ma150 and last > ma200, f"{last:.2f} vs {ma150:.2f}/{ma200:.2f}"),
        ("150MA above 200MA", ma150 > ma200, f"{ma150:.2f} vs {ma200:.2f}"),
        ("200MA rising for 1 month", bool(pd.notna(ma200_1m) and ma200 > ma200_1m), f"{ma200:.2f} vs {ma200_1m:.2f}" if pd.notna(ma200_1m) else "N/A"),
        ("50MA above 150MA and 200MA", ma50 > ma150 and ma50 > ma200, f"{ma50:.2f} / {ma150:.2f} / {ma200:.2f}"),
        ("Price above 50MA", last > ma50, f"{(last / ma50 - 1) * 100:.1f}% from MA50"),
        ("Price 30% above 52W low", last >= low52w * 1.3, f"{(last / low52w - 1) * 100:.0f}% from low"),
        ("Price within 25% of 52W high", last >= high52w * 0.75, f"{(last / high52w - 1) * 100:.1f}% from high"),
        ("Relative strength beats benchmark", rs126 is None or rs126 >= -5, "No benchmark" if rs126 is None else f"{rs126:.1f} pp excess 126D"),
    ]
    checks_df = pd.DataFrame(
        [{"check": label, "status": "pass" if passed else "fail", "evidence": evidence} for label, passed, evidence in checks]
    )
    fails = int((checks_df["status"] == "fail").sum())
    extension = (last / ma50 - 1) * 100 if ma50 else 0
    if fails:
        verdict = "PASS / trend template not ready"
    elif extension >= 25:
        verdict = "WATCH / extended above MA50"
    else:
        verdict = "WATCH / trend template passed, wait for valid base"

    summary = pd.DataFrame(
        [
            {"metric": "verdict", "value": verdict},
            {"metric": "passes", "value": int((checks_df["status"] == "pass").sum())},
            {"metric": "fails", "value": fails},
            {"metric": "last_close", "value": round(last, 2)},
            {"metric": "52w_high", "value": round(high52w, 2)},
            {"metric": "52w_low", "value": round(low52w, 2)},
            {"metric": "rs_126d_excess_pp", "value": None if rs126 is None else round(rs126, 2)},
        ]
    )
    return {"summary": summary, "checks": checks_df, "levels": data, "verdict": verdict}


def sepa_entry_plan(levels: pd.DataFrame) -> pd.DataFrame:
    if levels.empty:
        return pd.DataFrame()
    row = levels.dropna(subset=["ma50"]).tail(1)
    if row.empty:
        return pd.DataFrame()
    last = float(row["close"].iloc[0])
    ma50 = float(row["ma50"].iloc[0])
    pivot = round(max(last, levels["high"].tail(55).max()), 2)
    stop = round(min(ma50, pivot * 0.93), 2)
    return pd.DataFrame(
        [
            {"item": "pivot", "value": pivot, "note": "Hypothetical breakout pivot from recent range"},
            {"item": "buy_zone_high", "value": round(pivot * 1.05, 2), "note": "SEPA-style +5% max chase zone"},
            {"item": "stop", "value": stop, "note": "Research stop reference, not an order"},
            {"item": "risk_reward_to_15pct", "value": round((pivot * 1.15 - pivot) / max(pivot - stop, 0.01), 2), "note": "Needs 2R+ to stay interesting"},
        ]
    )

