from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SAMPLE_DIR = ROOT / "data" / "sample"
CONFIG_DIR = ROOT / "configs"


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


def load_prices() -> pd.DataFrame:
    df = pd.read_csv(SAMPLE_DIR / "us_stock_ohlcv.csv", parse_dates=["date"])
    return df.sort_values(["ticker", "date"]).reset_index(drop=True)


def load_crypto_prices() -> pd.DataFrame:
    df = pd.read_csv(SAMPLE_DIR / "crypto_ohlcv_hourly.csv", parse_dates=["datetime"])
    return df.sort_values(["symbol", "datetime"]).reset_index(drop=True)


def load_financials() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_DIR / "financial_metrics.csv", parse_dates=["quarter"])


def load_watchlist_notes() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_DIR / "watchlist_notes.csv")


def load_kronos_forecast() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_DIR / "kronos_forecast_sample.csv", parse_dates=["date"])


def load_paper_trades() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_DIR / "paper_trades_sample.csv", parse_dates=["entry_time", "exit_time"])
