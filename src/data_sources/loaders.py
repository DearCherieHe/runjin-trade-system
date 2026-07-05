from pathlib import Path

import pandas as pd

from src.data_sources.live_sources import (
    fetch_binance_hourly,
    fetch_yfinance_crypto_hourly,
    fetch_yfinance_prices,
    tiger_openapi_status,
)


ROOT = Path(__file__).resolve().parents[2]
SAMPLE_DIR = ROOT / "data" / "sample"
CONFIG_DIR = ROOT / "configs"
SOURCE_STATUS = {
    "us_equities": {"mode": "sample", "source": "sample", "message": "Offline sample data"},
    "crypto": {"mode": "sample", "source": "sample", "message": "Offline sample data"},
    "financials": {"mode": "sample", "source": "sample", "message": "Quarterly sample fundamentals"},
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
    return data_mode or "sample"


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
                raise RuntimeError("; ".join(live_errors)) from exc
            SOURCE_STATUS["crypto"] = {
                "mode": "sample_fallback",
                "source": "sample",
                "message": "Live sources unavailable: " + " | ".join(live_errors),
            }
    else:
        SOURCE_STATUS["crypto"] = {"mode": "sample", "source": "sample", "message": "Offline sample data"}
    return _sample_crypto_prices()


def load_financials() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_DIR / "financial_metrics.csv", parse_dates=["quarter"])


def load_watchlist_notes() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_DIR / "watchlist_notes.csv")


def load_kronos_forecast() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_DIR / "kronos_forecast_sample.csv", parse_dates=["date"])


def load_paper_trades() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_DIR / "paper_trades_sample.csv", parse_dates=["entry_time", "exit_time"])
