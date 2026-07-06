import os
from pathlib import Path

import pandas as pd

from src.data_sources.loaders import load_yaml


ROOT = Path(__file__).resolve().parents[2]
SAMPLE_DIR = ROOT / "data" / "sample"
CONFIG_DIR = ROOT / "configs"


def load_finance_mcp_config() -> dict:
    try:
        return load_yaml(CONFIG_DIR / "finance_mcp.yaml")
    except ImportError:
        return {
            "integration": {
                "mode": "sample_first",
                "optional_http_url_env": "FINANCE_MCP_HTTP_URL",
                "optional_research_csv_env": "FINANCE_MCP_RESEARCH_CSV",
            },
            "capabilities": [
                {"name": "market_news", "label": "Market news", "use_case": "Track narrative inflection, catalysts, and disagreement."},
                {"name": "macro_calendar", "label": "Macro calendar", "use_case": "Map CPI, rates, liquidity, and policy events against position risk."},
                {"name": "money_flow", "label": "Money flow", "use_case": "Watch sector rotation, ETF flow, and unusual turnover."},
                {"name": "index_constituents", "label": "Index and constituents", "use_case": "Understand benchmark inclusion, sector peers, and relative-strength groups."},
                {"name": "company_fundamentals", "label": "Company fundamentals", "use_case": "Cross-check revenue, margin, cash flow, valuation, and filings."},
                {"name": "technical_indicators", "label": "Technical indicators", "use_case": "Add KDJ and multi-indicator context around K-line decisions."},
                {"name": "china_market", "label": "China A/H market", "use_case": "Keep optional FinShare/OpenData/TuShare-style market coverage ready."},
                {"name": "crypto_market", "label": "Crypto market", "use_case": "Track BTC/ETH liquidity context for the paper bot."},
            ],
        }


def load_finance_mcp_capabilities() -> pd.DataFrame:
    config = load_finance_mcp_config()
    return pd.DataFrame(config.get("capabilities", []))


def _normalize_research(df: pd.DataFrame, source_label: str) -> pd.DataFrame:
    data = df.copy()
    required = ["date", "domain", "symbol", "title", "signal", "importance", "source", "action"]
    for col in required:
        if col not in data.columns:
            data[col] = ""
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["importance"] = pd.to_numeric(data["importance"], errors="coerce").fillna(0).astype(int)
    data["source"] = data["source"].replace("", source_label)
    return data[required].dropna(subset=["date"]).sort_values(["date", "importance"], ascending=[False, False])


def load_finance_mcp_research(data_mode=None):
    config = load_finance_mcp_config()
    integration = config.get("integration", {})
    csv_env = integration.get("optional_research_csv_env", "FINANCE_MCP_RESEARCH_CSV")
    http_env = integration.get("optional_http_url_env", "FINANCE_MCP_HTTP_URL")
    external_csv = os.getenv(csv_env, "").strip()
    http_url = os.getenv(http_env, "").strip()

    if external_csv and Path(external_csv).exists():
        df = pd.read_csv(external_csv)
        status = {
            "mode": "external_csv",
            "source": external_csv,
            "message": f"FinanceMCP-style research loaded from {csv_env}",
        }
        return _normalize_research(df, "finance_mcp_external_csv"), status

    sample = pd.read_csv(SAMPLE_DIR / "finance_mcp_research.csv")
    if http_url:
        status = {
            "mode": "mcp_ready_sample_fallback",
            "source": "sample",
            "message": f"{http_env} is configured; V0.1 keeps using sample rows until the HTTP adapter is enabled.",
        }
    else:
        status = {
            "mode": "sample",
            "source": "finance_mcp_sample",
            "message": "FinanceMCP-style research catalog is offline-ready. Set FINANCE_MCP_RESEARCH_CSV or FINANCE_MCP_HTTP_URL to connect external data.",
        }
    return _normalize_research(sample, "finance_mcp_sample"), status
