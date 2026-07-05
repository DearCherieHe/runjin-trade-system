import pandas as pd


TIMEFRAME_RULES = {
    "1H": None,
    "1D": None,
    "1W": "W-FRI",
    "1M": "ME",
    "1Q": "QE",
    "1Y": "YE",
}


def ensure_datetime(df, column="date"):
    data = df.copy()
    data[column] = pd.to_datetime(data[column], errors="coerce")
    if getattr(data[column].dt, "tz", None) is not None:
        data[column] = data[column].dt.tz_convert(None)
    data[column] = data[column].astype("datetime64[ns]")
    return data.dropna(subset=[column]).sort_values(column).reset_index(drop=True)


def resample_ohlcv(df, timeframe, date_col="date"):
    data = ensure_datetime(df, date_col)
    if timeframe in {"1H", "1D"}:
        return data.rename(columns={date_col: "date"}).reset_index(drop=True)
    rule = TIMEFRAME_RULES[timeframe]
    indexed = data.set_index(date_col)
    out = indexed.resample(rule).agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    out = out.dropna(subset=["open", "high", "low", "close"]).reset_index()
    return out.rename(columns={date_col: "date"})


def apply_asof(df, as_of):
    if as_of is None:
        return df
    data = ensure_datetime(df, "date")
    cutoff = pd.Timestamp(as_of)
    return data.loc[data["date"] <= cutoff].reset_index(drop=True)


def coverage_summary(df):
    if df.empty:
        return "No data"
    data = ensure_datetime(df, "date")
    start = data["date"].min().strftime("%Y-%m-%d")
    end = data["date"].max().strftime("%Y-%m-%d")
    return f"{start} -> {end} / {len(data):,} bars"
