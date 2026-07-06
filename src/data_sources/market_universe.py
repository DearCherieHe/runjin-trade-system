from pathlib import Path

import pandas as pd


MIN_MARKET_CAP_USD = 300_000_000

UNIVERSE_COLUMNS = [
    "ticker",
    "yahoo_ticker",
    "company",
    "market",
    "exchange",
    "currency",
    "market_cap_local",
    "market_cap_usd",
    "source",
]

DEFAULT_FX_TO_USD = {
    "USD": 1.0,
    "CNY": 0.14,
    "HKD": 0.128,
    "SGD": 0.74,
}

MARKET_DEFINITIONS = {
    "US": {"exchange": "NYSE/Nasdaq/AMEX", "currency": "USD", "suffix": ""},
    "A_SHARE_SH": {"exchange": "SSE", "currency": "CNY", "suffix": ".SS"},
    "A_SHARE_SZ": {"exchange": "SZSE", "currency": "CNY", "suffix": ".SZ"},
    "HK": {"exchange": "HKEX", "currency": "HKD", "suffix": ".HK"},
    "SG": {"exchange": "SGX", "currency": "SGD", "suffix": ".SI"},
}


def _empty_universe() -> pd.DataFrame:
    return pd.DataFrame(columns=UNIVERSE_COLUMNS)


def _normalize_ticker(value) -> str:
    return str(value).strip().upper()


def _yahoo_ticker(ticker: str, suffix: str) -> str:
    ticker = _normalize_ticker(ticker)
    if not suffix:
        return ticker
    if suffix == ".HK" and ticker.isdigit():
        return f"{ticker.zfill(4)}{suffix}"
    if suffix in {".SS", ".SZ"} and ticker.isdigit():
        return f"{ticker.zfill(6)}{suffix}"
    return ticker if ticker.endswith(suffix) else f"{ticker}{suffix}"


def seed_market_universe() -> pd.DataFrame:
    rows = [
        ["NVDA", "NVIDIA", "US", "NYSE/Nasdaq/AMEX", "USD", 4_000_000_000_000],
        ["MSFT", "Microsoft", "US", "NYSE/Nasdaq/AMEX", "USD", 3_800_000_000_000],
        ["AAPL", "Apple", "US", "NYSE/Nasdaq/AMEX", "USD", 3_000_000_000_000],
        ["TSLA", "Tesla", "US", "NYSE/Nasdaq/AMEX", "USD", 900_000_000_000],
        ["AMD", "Advanced Micro Devices", "US", "NYSE/Nasdaq/AMEX", "USD", 250_000_000_000],
        ["600519", "Kweichow Moutai", "A_SHARE_SH", "SSE", "CNY", 1_800_000_000_000],
        ["601318", "Ping An Insurance", "A_SHARE_SH", "SSE", "CNY", 850_000_000_000],
        ["000858", "Wuliangye Yibin", "A_SHARE_SZ", "SZSE", "CNY", 500_000_000_000],
        ["300750", "CATL", "A_SHARE_SZ", "SZSE", "CNY", 900_000_000_000],
        ["0700", "Tencent", "HK", "HKEX", "HKD", 3_500_000_000_000],
        ["9988", "Alibaba", "HK", "HKEX", "HKD", 1_800_000_000_000],
        ["3690", "Meituan", "HK", "HKEX", "HKD", 780_000_000_000],
        ["1299", "AIA", "HK", "HKEX", "HKD", 650_000_000_000],
        ["D05", "DBS Group", "SG", "SGX", "SGD", 130_000_000_000],
        ["O39", "OCBC", "SG", "SGX", "SGD", 75_000_000_000],
        ["U11", "UOB", "SG", "SGX", "SGD", 55_000_000_000],
        ["Z74", "Singtel", "SG", "SGX", "SGD", 50_000_000_000],
    ]
    data = []
    for ticker, company, market, exchange, currency, market_cap_local in rows:
        suffix = MARKET_DEFINITIONS[market]["suffix"]
        data.append(
            {
                "ticker": _normalize_ticker(ticker),
                "yahoo_ticker": _yahoo_ticker(ticker, suffix),
                "company": company,
                "market": market,
                "exchange": exchange,
                "currency": currency,
                "market_cap_local": float(market_cap_local),
                "market_cap_usd": float(market_cap_local) * DEFAULT_FX_TO_USD[currency],
                "source": "seed_universe",
            }
        )
    return pd.DataFrame(data, columns=UNIVERSE_COLUMNS)


def normalize_universe_frame(df: pd.DataFrame, market: str, source: str, fx_to_usd=None) -> pd.DataFrame:
    if df is None or df.empty:
        return _empty_universe()

    fx_to_usd = fx_to_usd or DEFAULT_FX_TO_USD
    definition = MARKET_DEFINITIONS.get(market, {})
    exchange = definition.get("exchange", market)
    currency = definition.get("currency", "USD")
    suffix = definition.get("suffix", "")

    aliases = {
        "ticker": ["ticker", "symbol", "code", "证券代码", "股票代码"],
        "yahoo_ticker": ["yahoo_ticker", "yahoo_symbol"],
        "company": ["company", "name", "short_name", "证券简称", "股票简称"],
        "exchange": ["exchange", "交易所"],
        "currency": ["currency", "币种"],
        "market_cap_local": ["market_cap_local", "market_cap", "total_market_cap", "市值", "总市值"],
        "market_cap_usd": ["market_cap_usd", "市值USD"],
    }
    lower_lookup = {str(col).lower(): col for col in df.columns}

    def pick(name, default=None):
        for candidate in aliases[name]:
            if candidate.lower() in lower_lookup:
                return df[lower_lookup[candidate.lower()]]
        return default

    ticker_series = pick("ticker")
    if ticker_series is None:
        return _empty_universe()

    result = pd.DataFrame()
    result["ticker"] = ticker_series.map(_normalize_ticker)
    yahoo_series = pick("yahoo_ticker")
    if yahoo_series is None:
        result["yahoo_ticker"] = result["ticker"].map(lambda value: _yahoo_ticker(value, suffix))
    else:
        result["yahoo_ticker"] = yahoo_series.map(_normalize_ticker)
    result["company"] = pick("company", result["ticker"]).fillna(result["ticker"]).astype(str)
    result["market"] = market
    result["exchange"] = pick("exchange", exchange)
    result["currency"] = pick("currency", currency)

    market_cap_local = pd.to_numeric(pick("market_cap_local"), errors="coerce") if pick("market_cap_local") is not None else pd.Series(dtype=float)
    market_cap_usd = pd.to_numeric(pick("market_cap_usd"), errors="coerce") if pick("market_cap_usd") is not None else pd.Series(dtype=float)
    if market_cap_usd.empty:
        result["market_cap_local"] = market_cap_local
        result["market_cap_usd"] = [
            cap * fx_to_usd.get(str(curr).upper(), 1.0) if pd.notna(cap) else None
            for cap, curr in zip(result["market_cap_local"], result["currency"])
        ]
    else:
        result["market_cap_usd"] = market_cap_usd
        result["market_cap_local"] = market_cap_local if not market_cap_local.empty else market_cap_usd
    result["source"] = source
    return result[UNIVERSE_COLUMNS].dropna(subset=["ticker", "yahoo_ticker"])


def load_configured_listing_csvs(config: dict, root: Path) -> pd.DataFrame:
    frames = []
    fx_to_usd = config.get("fx_to_usd", DEFAULT_FX_TO_USD)
    for market, market_config in config.get("markets", {}).items():
        csv_path = market_config.get("listing_csv")
        if not csv_path:
            continue
        path = Path(csv_path)
        if not path.is_absolute():
            path = root / path
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        frames.append(normalize_universe_frame(frame, market, str(path), fx_to_usd=fx_to_usd))
    if not frames:
        return _empty_universe()
    return pd.concat(frames, ignore_index=True)


def filter_market_cap(df: pd.DataFrame, min_market_cap_usd=MIN_MARKET_CAP_USD) -> pd.DataFrame:
    if df.empty:
        return df
    result = df.copy()
    result["market_cap_usd"] = pd.to_numeric(result["market_cap_usd"], errors="coerce")
    result = result.loc[result["market_cap_usd"] >= float(min_market_cap_usd)]
    return result.sort_values(["market", "market_cap_usd"], ascending=[True, False]).reset_index(drop=True)


def build_market_universe(config: dict, root: Path, data_mode="live_auto") -> pd.DataFrame:
    universe_config = config.get("market_universe") or config.get("free_sources", {}).get("market_universe", {})
    threshold = universe_config.get("min_market_cap_usd", MIN_MARKET_CAP_USD)
    configured = load_configured_listing_csvs(universe_config, root)

    if not configured.empty:
        return filter_market_cap(configured, threshold)
    if data_mode == "live":
        raise RuntimeError("No market-universe listing CSV is configured for live strict mode")
    return filter_market_cap(seed_market_universe(), threshold)
