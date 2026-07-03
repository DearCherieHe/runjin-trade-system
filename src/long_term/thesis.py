def get_ticker_profile(scored_notes, financials, ticker):
    profile = scored_notes.loc[scored_notes["ticker"] == ticker]
    if profile.empty:
        return None, financials.iloc[0:0]
    ticker_financials = financials.loc[financials["ticker"] == ticker].sort_values("quarter")
    return profile.iloc[0], ticker_financials


def latest_financial_snapshot(ticker_financials):
    if ticker_financials.empty:
        return {}
    latest = ticker_financials.iloc[-1]
    previous = ticker_financials.iloc[-2] if len(ticker_financials) > 1 else latest
    return {
        "revenue_growth_yoy": latest["revenue_growth_yoy"],
        "gross_margin": latest["gross_margin"],
        "operating_cash_flow_margin": latest["operating_cash_flow_margin"],
        "net_income_margin": latest["net_income_margin"],
        "guide_change": latest["guide_change"],
        "revenue_growth_delta": latest["revenue_growth_yoy"] - previous["revenue_growth_yoy"],
    }
