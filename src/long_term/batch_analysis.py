import pandas as pd


def batch_long_analysis(scored, financials, tickers):
    rows = []
    for ticker in tickers:
        profile = scored.loc[scored["ticker"] == ticker]
        if profile.empty:
            continue
        profile = profile.iloc[0]
        ticker_financials = financials.loc[financials["ticker"] == ticker].sort_values("quarter")
        latest = ticker_financials.iloc[-1] if not ticker_financials.empty else {}
        rows.append(
            {
                "ticker": ticker,
                "company": profile["company"],
                "score": profile["total_score"],
                "bucket": profile["bucket"],
                "revenue_growth_yoy": latest.get("revenue_growth_yoy"),
                "gross_margin": latest.get("gross_margin"),
                "ocf_margin": latest.get("operating_cash_flow_margin"),
                "key_risk": profile["invalidation"],
                "next_check": profile["growth_evidence"],
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values(["score", "revenue_growth_yoy"], ascending=False).reset_index(drop=True)


def analysis_progress_rows(tickers, status="queued"):
    return pd.DataFrame(
        [
            {"step": idx + 1, "ticker": ticker, "status": status}
            for idx, ticker in enumerate(tickers)
        ]
    )
