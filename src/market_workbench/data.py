from pathlib import Path

import pandas as pd

from src.ashare_workbench.sample_data import load_ashare_concepts, load_ashare_prices


ROOT = Path(__file__).resolve().parents[2]
SAMPLE_DIR = ROOT / "data" / "sample"


def load_hk_prices(path: Path | None = None) -> pd.DataFrame:
    source = path or SAMPLE_DIR / "hk_ohlcv.csv"
    df = pd.read_csv(source, dtype={"ticker": str})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["date", "ticker", "close"]).sort_values(["ticker", "date"]).reset_index(drop=True)


def load_market_concepts(path: Path | None = None) -> pd.DataFrame:
    source = path or SAMPLE_DIR / "market_concepts.csv"
    df = pd.read_csv(source, dtype={"ticker": str}).fillna("")
    return df


def _normalize_ohlcv(df: pd.DataFrame, ticker_col: str, market: str, asset_class: str, metadata: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    data = df.copy()
    data["ticker"] = data[ticker_col].astype(str)
    if "date" not in data.columns and "datetime" in data.columns:
        data["date"] = data["datetime"]
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    meta = metadata.loc[metadata["market"] == market].set_index("ticker").to_dict("index") if not metadata.empty else {}
    data["market"] = market
    data["asset_class"] = asset_class
    data["company"] = data["ticker"].map(lambda item: meta.get(item, {}).get("company", item))
    data["concept"] = data["ticker"].map(lambda item: meta.get(item, {}).get("concept", market))
    data["currency"] = data["ticker"].map(lambda item: meta.get(item, {}).get("currency", ""))
    if "amount" not in data.columns:
        data["amount"] = data["close"] * data["volume"]
    cols = ["date", "ticker", "market", "asset_class", "company", "concept", "currency", "open", "high", "low", "close", "volume", "amount"]
    return data[cols].dropna(subset=["date", "close"]).sort_values(["ticker", "date"]).reset_index(drop=True)


def build_market_workbench_data(us_prices: pd.DataFrame, crypto_prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata = load_market_concepts()
    ashare_prices = load_ashare_prices()
    ashare_meta = load_ashare_concepts().copy()
    ashare_meta["market"] = "A_SHARE"
    ashare_meta["asset_class"] = "equity"
    ashare_meta["currency"] = "CNY"
    ashare = _normalize_ohlcv(ashare_prices, "ticker", "A_SHARE", "equity", ashare_meta)
    us = _normalize_ohlcv(us_prices, "ticker", "US", "equity", metadata)
    hk = _normalize_ohlcv(load_hk_prices(), "ticker", "HK", "equity", metadata)
    crypto = crypto_prices.rename(columns={"symbol": "ticker"}).copy()
    crypto = _normalize_ohlcv(crypto, "ticker", "CRYPTO", "crypto", metadata)
    all_prices = pd.concat([ashare, us, hk, crypto], ignore_index=True)
    all_meta = pd.concat([metadata, ashare_meta], ignore_index=True, sort=False).drop_duplicates(["market", "ticker"], keep="first")
    return all_prices, all_meta
