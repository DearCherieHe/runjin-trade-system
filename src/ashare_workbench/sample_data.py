from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SAMPLE_DIR = ROOT / "data" / "sample"


def load_ashare_prices(path: Path | None = None) -> pd.DataFrame:
    source = path or SAMPLE_DIR / "ashare_ohlcv.csv"
    df = pd.read_csv(source, dtype={"ticker": str})
    df["ticker"] = df["ticker"].astype(str).str.zfill(6)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    numeric_cols = ["open", "high", "low", "close", "volume", "amount", "turnover_rate"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["date", "ticker", "close"]).sort_values(["ticker", "date"]).reset_index(drop=True)


def load_ashare_concepts(path: Path | None = None) -> pd.DataFrame:
    source = path or SAMPLE_DIR / "ashare_concepts.csv"
    df = pd.read_csv(source, dtype={"ticker": str}).fillna("")
    df["ticker"] = df["ticker"].astype(str).str.zfill(6)
    return df
