from src.data_sources.loaders import load_kronos_forecast


def get_research_forecast(ticker: str):
    data = load_kronos_forecast()
    return data.loc[data["ticker"] == ticker].copy()
