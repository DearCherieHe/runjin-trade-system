from __future__ import annotations

from typing import Any

import pandas as pd


def survivorship_audit_frame(raw_data: pd.DataFrame, spec: dict[str, Any]) -> pd.DataFrame:
    config = spec.get("survivorship_check", {}) or {}
    data_universe = str(config.get("data_universe", "current_listed_symbols")).strip() or "current_listed_symbols"
    includes_delisted = bool(config.get("includes_delisted", False))
    includes_bankrupt = bool(config.get("includes_bankrupt", includes_delisted))
    includes_acquired = bool(config.get("includes_acquired", includes_delisted))
    point_in_time_membership = bool(config.get("point_in_time_membership", False))
    low_price_bias = bool(config.get("low_price_bias", False))
    low_valuation_bias = bool(config.get("low_valuation_bias", False))
    distress_bias = bool(config.get("distress_bias", False))
    min_price = _min_price(raw_data)
    inferred_low_price = min_price is not None and min_price < float(config.get("low_price_threshold", 5))
    vulnerable_strategy = low_price_bias or low_valuation_bias or distress_bias or inferred_low_price

    rows = [
        {
            "check": "data_universe_declared",
            "status": "pass" if data_universe != "unknown" else "review",
            "value": data_universe,
            "required": "point-in-time universe preferred",
            "reason": "Backtests should disclose whether the database is current-listed only or survivorship-free.",
        },
        {
            "check": "includes_delisted_bankrupt_acquired",
            "status": _coverage_status(includes_delisted, includes_bankrupt, includes_acquired),
            "value": f"delisted={includes_delisted}, bankrupt={includes_bankrupt}, acquired={includes_acquired}",
            "required": "all true for survivorship-free equity tests",
            "reason": "A current-survivor-only database can overstate performance, especially for low-price or value-style strategies.",
        },
        {
            "check": "point_in_time_membership",
            "status": "pass" if point_in_time_membership else "review",
            "value": str(point_in_time_membership).lower(),
            "required": "true",
            "reason": "Universe membership should reflect what was tradable at each historical date, not what survives today.",
        },
        {
            "check": "strategy_survivorship_vulnerability",
            "status": _vulnerability_status(vulnerable_strategy, includes_delisted, point_in_time_membership),
            "value": f"low_price={low_price_bias or inferred_low_price}, low_valuation={low_valuation_bias}, distress={distress_bias}",
            "required": "vulnerable strategies need survivorship-free data",
            "reason": f"Minimum observed price={min_price:.2f}." if min_price is not None else "Price data unavailable for vulnerability inference.",
        },
    ]
    return pd.DataFrame(rows)


def _coverage_status(includes_delisted: bool, includes_bankrupt: bool, includes_acquired: bool) -> str:
    if includes_delisted and includes_bankrupt and includes_acquired:
        return "pass"
    if includes_delisted or includes_bankrupt or includes_acquired:
        return "review"
    return "review"


def _vulnerability_status(vulnerable: bool, includes_delisted: bool, point_in_time: bool) -> str:
    if not vulnerable:
        return "pass" if includes_delisted and point_in_time else "review"
    if includes_delisted and point_in_time:
        return "pass"
    return "fail"


def _min_price(raw_data: pd.DataFrame) -> float | None:
    if raw_data is None or raw_data.empty or "close" not in raw_data.columns:
        return None
    close = pd.to_numeric(raw_data["close"], errors="coerce").dropna()
    if close.empty:
        return None
    return float(close.min())
