from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def recent_regime_audit_frame(data: pd.DataFrame, equity_curve: pd.DataFrame, trades: pd.DataFrame, spec: dict[str, Any], stats: dict[str, Any]) -> pd.DataFrame:
    config = spec.get("regime_check", {}) or {}
    recent_years = float(config.get("recent_years", 3))
    min_recent_bars = int(config.get("min_recent_bars", 174))
    max_history_years_for_equal_weight = float(config.get("max_history_years_for_equal_weight", 5))
    max_recent_sharpe_decay = float(config.get("max_recent_sharpe_decay", 0.75))
    min_recent_sharpe = float(config.get("min_recent_sharpe", 0))
    current_costs_applied = bool(config.get("current_costs_applied_to_full_history", True))
    known_regime_breaks = _known_breaks(config.get("known_regime_breaks", ""))

    prepared = _prepare_equity(equity_curve)
    if prepared.empty:
        return pd.DataFrame(
            [
                {
                    "check": "recent_regime_window",
                    "status": "review",
                    "value": "N/A",
                    "threshold": "non-empty equity curve",
                    "reason": "No equity curve available for recent-regime review.",
                }
            ]
        )

    start = prepared["date"].min()
    end = prepared["date"].max()
    history_years = max((end - start).days / 365.25, 0)
    cutoff = end - pd.Timedelta(days=int(recent_years * 365.25))
    recent = prepared.loc[prepared["date"] >= cutoff].copy()
    early = prepared.loc[prepared["date"] < cutoff].copy()
    full_metrics = _equity_metrics(prepared)
    recent_metrics = _equity_metrics(recent)
    early_metrics = _equity_metrics(early)
    full_sharpe = _coerce_float(stats.get("Sharpe Ratio"))
    if full_sharpe is None:
        full_sharpe = full_metrics["sharpe"]

    rows = [
        {
            "check": "backtest_history_span",
            "status": "review" if history_years > max_history_years_for_equal_weight else "pass",
            "value": f"{history_years:.1f} years",
            "threshold": f"<= {max_history_years_for_equal_weight:.1f} years can be weighted evenly; older history needs regime context",
            "reason": "Long histories can mix different market structure, participant crowding, spreads, and survivor sets.",
        },
        {
            "check": "recent_regime_window",
            "status": "pass" if len(recent) >= min_recent_bars else "review",
            "value": f"{len(recent)} bars over last {recent_years:g} years",
            "threshold": f">= {min_recent_bars} recent bars",
            "reason": "Recent data deserves the heaviest weight because financial time series are often non-stationary.",
        },
        {
            "check": "recent_vs_full_performance",
            "status": _recent_status(full_sharpe, recent_metrics["sharpe"], recent_metrics["return_pct"], max_recent_sharpe_decay, min_recent_sharpe),
            "value": f"full_sharpe={_fmt(full_sharpe)}, recent_sharpe={_fmt(recent_metrics['sharpe'])}, recent_return={recent_metrics['return_pct']:.1f}%",
            "threshold": f"recent sharpe >= {min_recent_sharpe:g}; sharpe decay <= {max_recent_sharpe_decay:g}",
            "reason": "A strong full-period result is not enough if the recent regime no longer supports the edge.",
        },
        {
            "check": "early_performance_inflation",
            "status": _early_inflation_status(early_metrics["sharpe"], recent_metrics["sharpe"], early_metrics["return_pct"], recent_metrics["return_pct"]),
            "value": f"early_sharpe={_fmt(early_metrics['sharpe'])}, recent_sharpe={_fmt(recent_metrics['sharpe'])}, early_return={early_metrics['return_pct']:.1f}%, recent_return={recent_metrics['return_pct']:.1f}%",
            "threshold": "early edge should not dominate recent evidence",
            "reason": "Older backtest periods can look better because fewer funds competed, spreads differed, and missing delisted stocks distort early history.",
        },
        {
            "check": "historical_cost_stationarity",
            "status": "review" if current_costs_applied and history_years > max_history_years_for_equal_weight else "pass",
            "value": f"current_costs_applied_to_full_history={str(current_costs_applied).lower()}",
            "threshold": "cost model should be period-aware for long histories",
            "reason": "Applying today's cost assumptions to all historical years can make older returns unrealistic when spreads and liquidity regimes changed.",
        },
        {
            "check": "regime_change_disclosure",
            "status": "pass" if known_regime_breaks else "review",
            "value": ", ".join(known_regime_breaks) if known_regime_breaks else "none declared",
            "threshold": "declare known regulatory, market-structure, or macro regime breaks",
            "reason": "Decimalization, short-sale rule changes, crises, and liquidity regime shifts can invalidate one-model-fits-all assumptions.",
        },
        {
            "check": "nonstationarity_warning",
            "status": _nonstationary_status(prepared),
            "value": _nonstationary_value(prepared),
            "threshold": "recent volatility/return profile should not radically diverge from older history",
            "reason": "More data only improves confidence when the process is stable; markets often are not.",
        },
    ]
    return pd.DataFrame(rows)


def _prepare_equity(equity_curve: pd.DataFrame) -> pd.DataFrame:
    if equity_curve is None or equity_curve.empty:
        return pd.DataFrame()
    date_col = "Date" if "Date" in equity_curve.columns else "date" if "date" in equity_curve.columns else equity_curve.columns[0]
    equity_col = "Equity" if "Equity" in equity_curve.columns else equity_curve.columns[-1]
    prepared = equity_curve[[date_col, equity_col]].copy()
    prepared.columns = ["date", "equity"]
    prepared["date"] = pd.to_datetime(prepared["date"], errors="coerce")
    prepared["equity"] = pd.to_numeric(prepared["equity"], errors="coerce")
    return prepared.dropna(subset=["date", "equity"]).sort_values("date")


def _equity_metrics(frame: pd.DataFrame) -> dict[str, float | None]:
    if frame is None or len(frame) < 3:
        return {"return_pct": 0.0, "sharpe": None, "volatility_pct": None}
    equity = pd.to_numeric(frame["equity"], errors="coerce").dropna()
    if len(equity) < 3 or equity.iloc[0] == 0:
        return {"return_pct": 0.0, "sharpe": None, "volatility_pct": None}
    returns = equity.pct_change().dropna()
    total_return = float((equity.iloc[-1] / equity.iloc[0] - 1) * 100)
    if returns.empty or returns.std() == 0:
        return {"return_pct": total_return, "sharpe": None, "volatility_pct": 0.0}
    annualized_return = (1 + total_return / 100) ** (252 / max(len(equity), 1)) - 1
    annualized_vol = float(returns.std() * np.sqrt(252))
    sharpe = annualized_return / annualized_vol if annualized_vol > 0 else None
    return {
        "return_pct": total_return,
        "sharpe": float(sharpe) if sharpe is not None and np.isfinite(sharpe) else None,
        "volatility_pct": annualized_vol * 100,
    }


def _recent_status(full_sharpe: float | None, recent_sharpe: float | None, recent_return: float, max_decay: float, min_recent_sharpe: float) -> str:
    if recent_sharpe is None:
        return "review"
    if recent_sharpe < min_recent_sharpe or recent_return < -10:
        return "fail"
    if full_sharpe is not None and full_sharpe - recent_sharpe > max_decay:
        return "review"
    return "pass"


def _early_inflation_status(early_sharpe: float | None, recent_sharpe: float | None, early_return: float, recent_return: float) -> str:
    if early_sharpe is None or recent_sharpe is None:
        return "review"
    if early_sharpe - recent_sharpe > 1.5 and early_return > recent_return:
        return "fail"
    if early_sharpe - recent_sharpe > 0.75 or early_return > recent_return * 2:
        return "review"
    return "pass"


def _nonstationary_status(frame: pd.DataFrame) -> str:
    value = _nonstationary_ratio(frame)
    if value is None:
        return "review"
    if value >= 2.5 or value <= 0.4:
        return "review"
    return "pass"


def _nonstationary_value(frame: pd.DataFrame) -> str:
    value = _nonstationary_ratio(frame)
    return "N/A" if value is None else f"recent_vol / early_vol = {value:.2f}"


def _nonstationary_ratio(frame: pd.DataFrame) -> float | None:
    if frame is None or len(frame) < 60:
        return None
    midpoint = len(frame) // 2
    early = frame.iloc[:midpoint]["equity"].pct_change().dropna()
    recent = frame.iloc[midpoint:]["equity"].pct_change().dropna()
    if early.empty or recent.empty or early.std() == 0:
        return None
    return float(recent.std() / early.std())


def _known_breaks(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return []


def _coerce_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _fmt(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.3f}"
