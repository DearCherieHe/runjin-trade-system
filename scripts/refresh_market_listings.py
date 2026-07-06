from pathlib import Path
import argparse
import os
import sys
import tempfile

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CONFIG_PATH = ROOT / "configs" / "live_sources.yaml"
REQUIRED_COLUMNS = ["ticker", "company", "exchange", "currency", "market_cap_local", "market_cap_usd"]
DEFAULT_LISTING_PATHS = {
    "US": ROOT / "data/listings/us_listings.csv",
    "A_SHARE_SH": ROOT / "data/listings/a_share_sh_listings.csv",
    "A_SHARE_SZ": ROOT / "data/listings/a_share_sz_listings.csv",
    "HK": ROOT / "data/listings/hk_listings.csv",
    "SG": ROOT / "data/listings/sg_listings.csv",
}


def load_market_config():
    try:
        import yaml
    except ImportError:
        return {"markets": {market: {"listing_csv": str(path)} for market, path in DEFAULT_LISTING_PATHS.items()}}

    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    return config.get("free_sources", {}).get("market_universe", {})


def configured_listing_paths():
    universe_config = load_market_config()
    for market, market_config in universe_config.get("markets", {}).items():
        csv_path = market_config.get("listing_csv")
        if not csv_path:
            yield market, None
            continue
        path = Path(csv_path)
        if not path.is_absolute():
            path = ROOT / path
        yield market, path


def atomic_write_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        df.to_csv(temp_path, index=False)
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def validate_listing(path: Path):
    if path is None:
        return False, "no listing_csv configured"
    if not path.exists():
        return False, "file missing"
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        return False, f"read failed: {exc}"
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        return False, f"missing columns: {', '.join(missing)}"
    if df.empty:
        return False, "file is empty"
    return True, f"{len(df):,} rows"


def seed_missing_listing(market: str, path: Path):
    from src.data_sources.market_universe import seed_market_universe

    seed = seed_market_universe()
    frame = seed.loc[seed["market"] == market, REQUIRED_COLUMNS].copy()
    if frame.empty:
        return False, "no built-in seed rows for this market"
    atomic_write_csv(frame, path)
    return True, f"wrote {len(frame):,} seed rows"


def refresh_us_market_caps(path: Path):
    try:
        import yfinance as yf
    except ImportError:
        return False, "yfinance is not installed"

    if path is None or not path.exists():
        return False, "US listing CSV is missing"

    df = pd.read_csv(path)
    updated = 0
    for idx, ticker in df["ticker"].dropna().items():
        try:
            market_cap = yf.Ticker(str(ticker)).fast_info.get("market_cap")
        except Exception:
            market_cap = None
        if market_cap:
            df.loc[idx, "market_cap_local"] = float(market_cap)
            df.loc[idx, "market_cap_usd"] = float(market_cap)
            updated += 1

    if updated == 0:
        return False, "no US market caps refreshed; existing CSV was left unchanged"

    atomic_write_csv(df, path)
    return True, f"refreshed {updated:,} US market caps"


def main():
    parser = argparse.ArgumentParser(description="Validate or refresh local market listing CSV files.")
    parser.add_argument("--seed-if-missing", action="store_true", help="Create missing listing CSVs from the built-in seed universe.")
    parser.add_argument("--yfinance-us", action="store_true", help="Optionally refresh US market caps with yfinance.")
    args = parser.parse_args()

    us_path = None
    for market, path in configured_listing_paths():
        if market == "US":
            us_path = path
        ok, message = validate_listing(path)
        if not ok and args.seed_if_missing and path is not None:
            ok, message = seed_missing_listing(market, path)
        status = "OK" if ok else "WARN"
        print(f"{status} {market}: {message}")

    if args.yfinance_us:
        ok, message = refresh_us_market_caps(us_path)
        status = "OK" if ok else "WARN"
        print(f"{status} US yfinance refresh: {message}")


if __name__ == "__main__":
    main()
