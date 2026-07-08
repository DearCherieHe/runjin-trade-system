from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CostModel:
    name: str
    commission_rate: float
    min_commission: float
    notes: str


COMMISSION_MODELS = {
    "pct_only": CostModel("pct_only", 0.0010, 0.0, "Simple percent commission from strategy YAML."),
    "us_equity_basic": CostModel("us_equity_basic", 0.0005, 1.0, "Research estimate for US equities; broker schedules differ."),
    "a_share_basic": CostModel("a_share_basic", 0.00055, 5.0, "Research estimate with commission plus sell-side stamp duty approximation."),
    "hk_equity_basic": CostModel("hk_equity_basic", 0.0030, 0.0, "Research estimate for HK commission and duties."),
    "crypto_basic": CostModel("crypto_basic", 0.0010, 0.0, "Research estimate for spot crypto taker fee."),
}

SLIPPAGE_MODELS = {
    "close": "Execute at close with no extra slippage assumption.",
    "hl_mean": "Use the high-low midpoint as a conservative reference price.",
    "hl_mean_gap_guard": "Use high-low midpoint and flag unusually large gaps for review.",
}


def resolve_commission_rate(model_name: str, fallback_rate: float) -> float:
    model = COMMISSION_MODELS.get(model_name)
    if model is None or model_name == "pct_only":
        return float(fallback_rate)
    return float(model.commission_rate)


def estimate_commission(model_name: str, trade_value: float, side: str = "buy") -> float:
    model = COMMISSION_MODELS.get(model_name, COMMISSION_MODELS["pct_only"])
    commission = abs(float(trade_value)) * model.commission_rate
    if side == "sell" and model_name == "a_share_basic":
        commission += abs(float(trade_value)) * 0.0005
    return max(commission, model.min_commission if trade_value else 0.0)


def describe_execution_assumptions(
    commission_model: str,
    slippage_model: str,
    trade_on_close: bool,
    position_model: str,
    benchmark: str,
) -> pd.DataFrame:
    cost_model = COMMISSION_MODELS.get(commission_model, COMMISSION_MODELS["pct_only"])
    return pd.DataFrame(
        [
            {"setting": "commission_model", "value": commission_model, "description": cost_model.notes},
            {"setting": "slippage_model", "value": slippage_model, "description": SLIPPAGE_MODELS.get(slippage_model, "Unknown slippage model")},
            {
                "setting": "trade_on_close",
                "value": str(bool(trade_on_close)).lower(),
                "description": "False means signals generated from the current bar are executed on the next bar, reducing look-ahead risk.",
            },
            {"setting": "position_model", "value": position_model, "description": "No leverage; capped by strategy position_size."},
            {"setting": "benchmark", "value": benchmark or "asset buy & hold", "description": "Used for ABU-style comparison metrics when available."},
        ]
    )


def slippage_reference(ohlcv: pd.DataFrame, model_name: str = "hl_mean_gap_guard", gap_threshold: float = 0.08) -> pd.DataFrame:
    data = ohlcv.copy()
    if data.empty:
        return data
    data["prev_close"] = data["close"].shift(1)
    if model_name in {"hl_mean", "hl_mean_gap_guard"}:
        data["execution_reference"] = (data["high"] + data["low"]) / 2
    else:
        data["execution_reference"] = data["close"]
    data["gap_pct"] = data["open"] / data["prev_close"] - 1
    data["gap_guard_flag"] = False
    if model_name == "hl_mean_gap_guard":
        data["gap_guard_flag"] = data["gap_pct"].abs() > float(gap_threshold)
    data["slippage_bps"] = np.where(
        data["close"].abs() > 0,
        (data["execution_reference"] / data["close"] - 1).abs() * 10000,
        0,
    )
    return data


def transaction_cost_audit_frame(spec: dict, stats: dict, metrics: dict, trades: pd.DataFrame) -> pd.DataFrame:
    config = spec.get("transaction_cost_check", {}) or {}
    commission_bps = float(spec.get("commission", 0)) * 10000
    half_spread_bps = float(config.get("half_spread_bps", 5))
    market_impact_bps = float(config.get("market_impact_bps", 0))
    latency_slippage_bps = float(config.get("latency_slippage_bps", 1))
    max_per_side_cost_bps = float(config.get("max_per_side_cost_bps", 20))
    max_annual_cost_drag_pct = float(config.get("max_annual_cost_drag_pct", 6))
    min_avg_trade_edge_multiple = float(config.get("min_avg_trade_edge_multiple", 2))

    per_side_cost_bps = commission_bps + half_spread_bps + market_impact_bps + latency_slippage_bps
    round_trip_cost_bps = per_side_cost_bps * 2
    round_trip_cost_pct = round_trip_cost_bps / 100
    trade_count = int(stats.get("# Trades", 0) or 0)
    trades_per_year = float(metrics.get("trades_per_year") or 0)
    annual_cost_drag_pct = round_trip_cost_pct * trades_per_year
    total_cost_drag_pct = round_trip_cost_pct * trade_count
    return_pct = _coerce_float(stats.get("Return [%]")) or 0
    sharpe = _coerce_float(stats.get("Sharpe Ratio"))
    annual_vol_pct = _coerce_float(stats.get("Volatility (Ann.) [%]"))
    avg_trade_pct = _coerce_float(stats.get("Avg. Trade [%]"))
    if avg_trade_pct is None and trades is not None and not trades.empty and "ReturnPct" in trades.columns:
        avg_trade_pct = float(pd.to_numeric(trades["ReturnPct"], errors="coerce").mean())

    edge_multiple = None
    avg_trade_after_cost = None
    if avg_trade_pct is not None and round_trip_cost_pct > 0:
        edge_multiple = avg_trade_pct / round_trip_cost_pct
        avg_trade_after_cost = avg_trade_pct - round_trip_cost_pct
    cost_adjusted_return = return_pct - total_cost_drag_pct
    cost_adjusted_sharpe = None
    if sharpe is not None and annual_vol_pct is not None and annual_vol_pct > 0:
        cost_adjusted_sharpe = sharpe - annual_cost_drag_pct / annual_vol_pct

    return pd.DataFrame(
        [
            {
                "check": "per_side_transaction_cost",
                "status": "pass" if per_side_cost_bps <= max_per_side_cost_bps else "review",
                "actual": f"{per_side_cost_bps:.1f} bps",
                "threshold": f"<= {max_per_side_cost_bps:.1f} bps",
                "reason": f"commission={commission_bps:.1f}, half_spread={half_spread_bps:.1f}, impact={market_impact_bps:.1f}, latency_slippage={latency_slippage_bps:.1f} bps.",
            },
            {
                "check": "round_trip_transaction_cost",
                "status": "pass" if round_trip_cost_bps <= max_per_side_cost_bps * 2 else "review",
                "actual": f"{round_trip_cost_bps:.1f} bps",
                "threshold": f"<= {max_per_side_cost_bps * 2:.1f} bps",
                "reason": "A complete buy-then-sell cycle counts as two separate transactions.",
            },
            {
                "check": "annual_cost_drag",
                "status": "pass" if annual_cost_drag_pct <= max_annual_cost_drag_pct else "fail",
                "actual": f"{annual_cost_drag_pct:.2f}%",
                "threshold": f"<= {max_annual_cost_drag_pct:.2f}%",
                "reason": f"Estimated from {trades_per_year:.1f} trades/year and {round_trip_cost_bps:.1f} bps round-trip cost.",
            },
            {
                "check": "average_trade_edge_vs_cost",
                "status": _edge_status(edge_multiple, min_avg_trade_edge_multiple),
                "actual": f"{edge_multiple:.2f}x" if edge_multiple is not None else "N/A",
                "threshold": f">= {min_avg_trade_edge_multiple:.2f}x",
                "reason": f"Avg trade={avg_trade_pct:.3f}% and avg trade after cost={avg_trade_after_cost:.3f}%." if avg_trade_pct is not None and avg_trade_after_cost is not None else "No average trade edge is available.",
            },
            {
                "check": "cost_adjusted_return_proxy",
                "status": "pass" if cost_adjusted_return > 0 else "fail",
                "actual": f"{cost_adjusted_return:.2f}%",
                "threshold": "> 0%",
                "reason": f"Conservative return proxy subtracting {total_cost_drag_pct:.2f}% total estimated round-trip drag from reported return.",
            },
            {
                "check": "cost_adjusted_sharpe_proxy",
                "status": _cost_adjusted_sharpe_status(cost_adjusted_sharpe),
                "actual": f"{cost_adjusted_sharpe:.2f}" if cost_adjusted_sharpe is not None else "N/A",
                "threshold": ">= 0.5 preferred",
                "reason": "Conservative Sharpe proxy after annualized cost drag. This is a sensitivity test, not a replacement for broker-accurate fills.",
            },
        ]
    )


def _edge_status(edge_multiple: float | None, threshold: float) -> str:
    if edge_multiple is None:
        return "review"
    if edge_multiple >= threshold:
        return "pass"
    if edge_multiple >= 1:
        return "review"
    return "fail"


def _cost_adjusted_sharpe_status(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "review"
    if value >= 0.5:
        return "pass"
    if value >= 0:
        return "review"
    return "fail"


def _coerce_float(value):
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
