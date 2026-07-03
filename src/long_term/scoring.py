SCORE_COLUMNS = [
    "narrative_space",
    "revenue_acceleration",
    "margin_quality",
    "cash_flow",
    "value_chain_position",
    "market_disagreement",
    "valuation_tolerance",
]


def score_bucket(total_score: int) -> str:
    if total_score >= 35:
        return "Deep research candidate"
    if total_score >= 28:
        return "Observation pool"
    return "Low priority"


def build_score_table(notes_df):
    scored = notes_df.copy()
    scored["total_score"] = scored[SCORE_COLUMNS].sum(axis=1)
    scored["bucket"] = scored["total_score"].map(score_bucket)
    scored["score_label"] = scored["total_score"].astype(str) + " / 35"
    return scored.sort_values(["total_score", "ticker"], ascending=[False, True]).reset_index(drop=True)
