from pathlib import Path

import pandas as pd

from src.data_sources.live_sources import (
    fetch_binance_hourly,
    fetch_yfinance_crypto_hourly,
    fetch_yfinance_financials,
    fetch_yfinance_prices,
    tiger_openapi_status,
)
from src.data_sources.market_universe import build_market_universe


ROOT = Path(__file__).resolve().parents[2]
SAMPLE_DIR = ROOT / "data" / "sample"
CONFIG_DIR = ROOT / "configs"
SOURCE_STATUS = {
    "us_equities": {"mode": "live_pending", "source": "yfinance", "message": "Waiting for live market data"},
    "market_universe": {"mode": "live_pending", "source": "configured_listings", "message": "Waiting for market universe"},
    "crypto": {"mode": "live_pending", "source": "binance_public_rest", "message": "Waiting for live crypto data"},
    "financials": {"mode": "live_pending", "source": "yfinance", "message": "Waiting for live quarterly fundamentals"},
    "forecast": {"mode": "disabled", "source": "none", "message": "Local sample forecast disabled in live mode"},
    "china_market": {"mode": "candidate", "source": "finshare/opendatatools/tushare", "message": "Optional adapters registered"},
    "broker": {"mode": "placeholder", "source": "tiger_openapi", "message": "Waiting for Tiger credentials"},
}


def load_yaml(path: Path) -> dict:
    try:
        import yaml

        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)
    except ImportError:
        if path.name == "watchlist.yaml":
            return _fallback_watchlist()
        if path.name == "risk_rules.yaml":
            return _fallback_risk_rules()
        if path.name == "live_sources.yaml":
            return _fallback_live_sources()
        raise


def _fallback_watchlist() -> dict:
    return {
        "watchlist": [
            {"ticker": "NVDA", "company": "NVIDIA", "tags": ["AI", "semiconductors", "data center"]},
            {"ticker": "TSLA", "company": "Tesla", "tags": ["EV", "autonomy", "energy storage"]},
            {"ticker": "SNDK", "company": "Sandisk", "tags": ["memory", "storage", "AI infrastructure"]},
            {"ticker": "PLTR", "company": "Palantir", "tags": ["AI software", "government", "enterprise"]},
            {"ticker": "AMD", "company": "Advanced Micro Devices", "tags": ["semiconductors", "AI accelerators", "CPU"]},
            {"ticker": "AVGO", "company": "Broadcom", "tags": ["semiconductors", "networking", "infrastructure software"]},
            {"ticker": "TSM", "company": "Taiwan Semiconductor Manufacturing", "tags": ["foundry", "advanced nodes", "AI supply chain"]},
            {"ticker": "ARM", "company": "Arm Holdings", "tags": ["chip IP", "mobile", "edge AI"]},
            {"ticker": "SOUN", "company": "SoundHound AI", "tags": ["voice AI", "automotive", "restaurants"]},
            {"ticker": "CRWV", "company": "CoreWeave", "tags": ["AI cloud", "GPUs", "data center"]},
        ]
    }


def _fallback_risk_rules() -> dict:
    return {
        "capital": {"starting_cash": 100000, "max_position_pct": 0.12, "no_leverage": True},
        "risk_limits": {
            "max_daily_loss_pct": 0.03,
            "max_drawdown_pct": 0.12,
            "per_trade_stop_loss_pct": 0.06,
            "min_cash_pct": 0.25,
        },
        "strategies": {
            "us_stock_daily": {
                "name": "Trend following",
                "fast_ma": 20,
                "slow_ma": 60,
                "volatility_window": 20,
                "max_annualized_volatility": 0.75,
            },
            "crypto_hourly": {
                "name": "Mean reversion",
                "rsi_window": 14,
                "lower_rsi": 35,
                "upper_rsi": 65,
                "bollinger_window": 20,
                "bollinger_std": 2,
            },
        },
    }


def _fallback_live_sources() -> dict:
    return {
        "mode": {"default": "live_auto", "fallback": "sample"},
        "free_sources": {
            "market_universe": {
                "min_market_cap_usd": 300000000,
                "top_symbols_per_market": 3000,
                "fx_to_usd": {"USD": 1.0, "CNY": 0.14, "HKD": 0.128, "SGD": 0.74},
                "markets": {
                    "US": {"exchange": "NYSE/Nasdaq/AMEX", "currency": "USD", "listing_csv": "data/listings/us_listings.csv"},
                    "A_SHARE_SH": {"exchange": "SSE", "currency": "CNY", "listing_csv": "data/listings/a_share_sh_listings.csv"},
                    "A_SHARE_SZ": {"exchange": "SZSE", "currency": "CNY", "listing_csv": "data/listings/a_share_sz_listings.csv"},
                    "HK": {"exchange": "HKEX", "currency": "HKD", "listing_csv": "data/listings/hk_listings.csv"},
                    "SG": {"exchange": "SGX", "currency": "SGD", "listing_csv": "data/listings/sg_listings.csv"},
                },
            },
            "crypto": {"symbols": {"BTC-USD": "BTCUSDT", "ETH-USD": "ETHUSDT"}},
        },
    }


def load_watchlist_config() -> dict:
    return load_yaml(CONFIG_DIR / "watchlist.yaml")


def load_risk_rules() -> dict:
    return load_yaml(CONFIG_DIR / "risk_rules.yaml")


def load_live_sources_config() -> dict:
    return load_yaml(CONFIG_DIR / "live_sources.yaml")


def get_data_source_status() -> dict:
    SOURCE_STATUS["broker"] = {"mode": "placeholder", **tiger_openapi_status()}
    return SOURCE_STATUS.copy()


def _normalize_mode(data_mode):
    return data_mode or "live"


def _sample_prices() -> pd.DataFrame:
    df = pd.read_csv(SAMPLE_DIR / "us_stock_ohlcv.csv", parse_dates=["date"])
    return df.sort_values(["ticker", "date"]).reset_index(drop=True)


def load_prices(data_mode=None) -> pd.DataFrame:
    mode = _normalize_mode(data_mode)
    if mode in {"live", "live_auto"}:
        tickers = [item["ticker"] for item in load_watchlist_config()["watchlist"]]
        try:
            df = fetch_yfinance_prices(tickers)
            SOURCE_STATUS["us_equities"] = {
                "mode": mode,
                "source": "yfinance",
                "message": "Live/near-live Yahoo Finance research feed",
            }
            return df.sort_values(["ticker", "date"]).reset_index(drop=True)
        except Exception as exc:
            if mode == "live":
                raise
            SOURCE_STATUS["us_equities"] = {
                "mode": "sample_fallback",
                "source": "sample",
                "message": f"Live source unavailable: {exc}",
            }
    else:
        SOURCE_STATUS["us_equities"] = {"mode": "sample", "source": "sample", "message": "Offline sample data"}
    return _sample_prices()


def _sample_crypto_prices() -> pd.DataFrame:
    df = pd.read_csv(SAMPLE_DIR / "crypto_ohlcv_hourly.csv", parse_dates=["datetime"])
    return df.sort_values(["symbol", "datetime"]).reset_index(drop=True)


def _empty_crypto_prices() -> pd.DataFrame:
    return pd.DataFrame(columns=["datetime", "symbol", "open", "high", "low", "close", "volume"])


def load_crypto_prices(data_mode=None) -> pd.DataFrame:
    mode = _normalize_mode(data_mode)
    if mode in {"live", "live_auto"}:
        live_errors = []
        try:
            config = load_live_sources_config()
            symbol_map = config["free_sources"]["crypto"]["symbols"]
            df = fetch_binance_hourly(symbol_map)
            SOURCE_STATUS["crypto"] = {
                "mode": mode,
                "source": "binance_public_rest",
                "message": "Public Binance hourly klines",
            }
            return df.sort_values(["symbol", "datetime"]).reset_index(drop=True)
        except Exception as exc:
            live_errors.append(f"binance_public_rest: {exc}")
        try:
            config = load_live_sources_config()
            symbols = list(config["free_sources"]["crypto"]["symbols"].keys())
            df = fetch_yfinance_crypto_hourly(symbols)
            SOURCE_STATUS["crypto"] = {
                "mode": mode,
                "source": "yfinance_crypto",
                "message": "Yahoo Finance crypto hourly fallback",
            }
            return df.sort_values(["symbol", "datetime"]).reset_index(drop=True)
        except Exception as exc:
            live_errors.append(f"yfinance_crypto: {exc}")
            if mode == "live":
                SOURCE_STATUS["crypto"] = {
                    "mode": "unavailable",
                    "source": "none",
                    "message": "Crypto live sources unavailable: " + " | ".join(live_errors),
                }
                return _empty_crypto_prices()
            SOURCE_STATUS["crypto"] = {
                "mode": "sample_fallback",
                "source": "sample",
                "message": "Live sources unavailable: " + " | ".join(live_errors),
            }
    else:
        SOURCE_STATUS["crypto"] = {"mode": "sample", "source": "sample", "message": "Offline sample data"}
    return _sample_crypto_prices()


def load_financials(data_mode=None) -> pd.DataFrame:
    mode = _normalize_mode(data_mode)
    if mode in {"live", "live_auto"}:
        tickers = [item["ticker"] for item in load_watchlist_config()["watchlist"]]
        try:
            df = fetch_yfinance_financials(tickers)
            SOURCE_STATUS["financials"] = {
                "mode": mode,
                "source": "yfinance",
                "message": "Live quarterly fundamentals from Yahoo Finance",
            }
            return df.sort_values(["ticker", "quarter"]).reset_index(drop=True)
        except Exception as exc:
            if mode == "live":
                raise
            SOURCE_STATUS["financials"] = {
                "mode": "sample_fallback",
                "source": "sample",
                "message": f"Live fundamentals unavailable: {exc}",
            }
    else:
        SOURCE_STATUS["financials"] = {"mode": "sample", "source": "sample", "message": "Quarterly sample fundamentals"}
    return pd.read_csv(SAMPLE_DIR / "financial_metrics.csv", parse_dates=["quarter"])


def load_market_universe(data_mode=None) -> pd.DataFrame:
    mode = _normalize_mode(data_mode)
    config = load_live_sources_config()
    try:
        df = build_market_universe(config, ROOT, data_mode=mode)
        source = "configured_listings" if not df.empty and not (df["source"] == "seed_universe").all() else "seed_universe"
        message = "US, A-share, HK, and SG universe filtered at USD 300M market cap and top 3000 per market"
        if source == "seed_universe" and mode == "live":
            message = "Live strict has no configured listing CSV; using built-in seed universe so the workspace can start"
        SOURCE_STATUS["market_universe"] = {
            "mode": mode if source != "seed_universe" else "sample_seed",
            "source": source,
            "message": message,
        }
        return df
    except Exception as exc:
        if mode == "live":
            raise
        df = build_market_universe(config, ROOT, data_mode="sample")
        SOURCE_STATUS["market_universe"] = {
            "mode": "sample_seed",
            "source": "seed_universe",
            "message": f"Configured universe unavailable: {exc}",
        }
        return df


def load_watchlist_notes() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_DIR / "watchlist_notes.csv")


def load_future_industry_map() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_DIR / "future_industry_map.csv")


def load_kronos_forecast(data_mode=None) -> pd.DataFrame:
    mode = _normalize_mode(data_mode)
    if mode in {"live", "live_auto"}:
        SOURCE_STATUS["forecast"] = {
            "mode": "disabled",
            "source": "none",
            "message": "Local sample forecast disabled while the app runs on live data",
        }
        return pd.DataFrame(columns=["date", "ticker", "predicted_close", "lower_band", "upper_band"])
    return pd.read_csv(SAMPLE_DIR / "kronos_forecast_sample.csv", parse_dates=["date"])


def load_paper_trades() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_DIR / "paper_trades_sample.csv", parse_dates=["entry_time", "exit_time"])
