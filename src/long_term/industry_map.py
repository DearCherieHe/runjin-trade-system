from __future__ import annotations

import pandas as pd


EDGE_COLUMNS = ["upstream_tickers", "peer_tickers", "downstream_tickers"]
MOAT_COLUMNS = ["monopoly_score", "gross_margin_power", "irreplaceability", "ten_year_optionality"]


def build_future_profile(industry_map: pd.DataFrame, scored: pd.DataFrame, ticker: str) -> dict:
    row = _ticker_row(industry_map, ticker)
    if row.empty:
        return {}
    record = row.iloc[0].to_dict()
    scored_row = scored.loc[scored["ticker"] == ticker] if scored is not None and not scored.empty and "ticker" in scored.columns else pd.DataFrame()
    if not scored_row.empty:
        record.update({f"long_{key}": value for key, value in scored_row.iloc[0].to_dict().items()})
    record["moat_total"] = sum(float(record.get(col, 0) or 0) for col in MOAT_COLUMNS)
    record["moat_label"] = _moat_label(record["moat_total"])
    return record


def industry_layer_scores(industry_map: pd.DataFrame, ticker: str) -> pd.DataFrame:
    row = _ticker_row(industry_map, ticker)
    if row.empty:
        return pd.DataFrame(columns=["mega_theme", "chain_layer", "avg_moat", "companies", "tickers", "selected"])
    theme = row.iloc[0]["mega_theme"]
    view = industry_map.loc[industry_map["mega_theme"] == theme].copy()
    if view.empty:
        return pd.DataFrame(columns=["mega_theme", "chain_layer", "avg_moat", "companies", "tickers", "selected"])
    view["moat_total"] = view[MOAT_COLUMNS].apply(pd.to_numeric, errors="coerce").sum(axis=1)
    grouped = (
        view.groupby(["mega_theme", "chain_layer"], as_index=False)
        .agg(avg_moat=("moat_total", "mean"), companies=("company", "count"), tickers=("ticker", lambda values: "; ".join(values)))
        .sort_values("avg_moat", ascending=False)
    )
    grouped["selected"] = grouped["chain_layer"].eq(row.iloc[0]["chain_layer"])
    return grouped


def relationship_comparison(industry_map: pd.DataFrame, scored: pd.DataFrame, ticker: str) -> pd.DataFrame:
    row = _ticker_row(industry_map, ticker)
    if row.empty:
        return pd.DataFrame(columns=["relationship", "ticker", "company", "mega_theme", "chain_layer", "moat_total", "score_label", "why_it_can_10x"])
    records = []
    base = row.iloc[0]
    records.append(_comparison_record(base, "selected"))
    for column, relationship in [
        ("upstream_tickers", "upstream"),
        ("peer_tickers", "peer"),
        ("downstream_tickers", "downstream"),
    ]:
        for related in _split_tickers(base.get(column, "")):
            related_row = _ticker_row(industry_map, related)
            if not related_row.empty:
                records.append(_comparison_record(related_row.iloc[0], relationship))
            else:
                records.append({"relationship": relationship, "ticker": related, "company": related, "mega_theme": "", "chain_layer": "Not in map yet", "moat_total": None, "score_label": "", "why_it_can_10x": "Add profile row to compare this company."})
    result = pd.DataFrame(records)
    if scored is not None and not scored.empty and "ticker" in scored.columns:
        score_cols = ["ticker", "score_label", "bucket", "total_score"]
        result = result.drop(columns=[col for col in ["score_label"] if col in result.columns], errors="ignore").merge(
            scored[[col for col in score_cols if col in scored.columns]],
            on="ticker",
            how="left",
        )
    if "score_label" not in result.columns:
        result["score_label"] = ""
    return result


def future_theme_summary(industry_map: pd.DataFrame) -> pd.DataFrame:
    if industry_map.empty:
        return pd.DataFrame(columns=["mega_theme", "companies", "avg_moat", "top_layers", "top_tickers"])
    view = industry_map.copy()
    view["moat_total"] = view[MOAT_COLUMNS].apply(pd.to_numeric, errors="coerce").sum(axis=1)
    grouped = (
        view.groupby("mega_theme", as_index=False)
        .agg(
            companies=("ticker", "count"),
            avg_moat=("moat_total", "mean"),
            top_layers=("chain_layer", lambda values: "; ".join(pd.Series(values).value_counts().head(3).index)),
            top_tickers=("ticker", lambda values: "; ".join(values.head(5))),
        )
        .sort_values("avg_moat", ascending=False)
    )
    return grouped


def _comparison_record(row: pd.Series, relationship: str) -> dict:
    moat_total = sum(float(row.get(col, 0) or 0) for col in MOAT_COLUMNS)
    return {
        "relationship": relationship,
        "ticker": row.get("ticker", ""),
        "company": row.get("company", ""),
        "mega_theme": row.get("mega_theme", ""),
        "industry": row.get("industry", ""),
        "chain_layer": row.get("chain_layer", ""),
        "chain_role": row.get("chain_role", ""),
        "moat_total": moat_total,
        "monopoly_score": row.get("monopoly_score", None),
        "gross_margin_power": row.get("gross_margin_power", None),
        "irreplaceability": row.get("irreplaceability", None),
        "ten_year_optionality": row.get("ten_year_optionality", None),
        "why_it_can_10x": row.get("why_it_can_10x", ""),
        "key_questions": row.get("key_questions", ""),
    }


def _ticker_row(industry_map: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if industry_map is None or industry_map.empty or "ticker" not in industry_map.columns:
        return pd.DataFrame()
    return industry_map.loc[industry_map["ticker"].astype(str).str.upper() == str(ticker).upper()]


def _split_tickers(value) -> list[str]:
    if pd.isna(value):
        return []
    return [item.strip().upper() for item in str(value).replace(",", ";").split(";") if item.strip()]


def _moat_label(total: float) -> str:
    if total >= 18:
        return "Potential monopoly layer"
    if total >= 15:
        return "Strong value-chain position"
    if total >= 12:
        return "Watch for proof"
    return "Speculative or cyclical"
