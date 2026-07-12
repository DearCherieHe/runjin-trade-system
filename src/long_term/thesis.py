HOLD_OBSERVATION_CONDITIONS = [
    "行业长期需求没有破。",
    "公司仍是核心供应商。",
    "财报没有连续恶化。",
    "技术路线没有被替代。",
    "估值已经反映悲观预期。",
    "周线/月线没有放量破长期平台。",
]

EXIT_OBSERVATION_CONDITIONS = [
    "需求逻辑被证伪。",
    "客户丢失或产品落后。",
    "毛利率长期坍塌。",
    "财务恶化。",
    "同行明显更强，而它只是跟涨概念。",
    "股价跌破长期平台后反抽无力。",
]

TENBAGGER_DISCOVERY_FRAMEWORK = [
    "产业趋势向上",
    "公司位置靠前",
    "财报逐步验证",
    "估值有安全边际",
    "月/周线开始止跌放量",
]


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
