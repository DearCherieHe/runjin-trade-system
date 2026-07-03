from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = ROOT / "data" / "sample"
CONFIG_PATH = ROOT / "configs" / "watchlist.yaml"


PROFILES = {
    "NVDA": (920, 0.0023, 0.032, 4.8),
    "TSLA": (245, 0.0009, 0.035, 3.7),
    "SNDK": (58, 0.0014, 0.038, 3.9),
    "PLTR": (28, 0.0018, 0.041, 4.4),
    "AMD": (165, 0.0011, 0.034, 3.8),
    "AVGO": (1550, 0.0017, 0.028, 4.5),
    "TSM": (175, 0.0012, 0.026, 4.1),
    "ARM": (135, 0.0015, 0.036, 3.6),
    "SOUN": (6.5, 0.0022, 0.058, 2.9),
    "CRWV": (42, 0.0025, 0.052, 3.3),
}


NOTES = {
    "NVDA": (
        "AI compute platform and data-center acceleration",
        "Data center demand keeps pulling forward accelerator, networking, and software attach.",
        "Blackwell ramp, sovereign AI, networking attach",
        "Hyperscaler capex digestion or margin compression",
        "Add only after earnings confirms demand and valuation cools on volatility.",
    ),
    "TSLA": (
        "Autonomy, EV scale, energy storage, and robot optionality",
        "Energy storage growth offsets EV pricing pressure while autonomy narrative remains unresolved.",
        "FSD milestones, energy deployments, lower-cost vehicle",
        "EV gross margin fails to stabilize or autonomy slips without evidence",
        "Build slowly; require operating margin stabilization.",
    ),
    "SNDK": (
        "AI storage and memory-cycle recovery",
        "Flash demand can benefit from AI storage growth and cyclical pricing recovery.",
        "NAND price recovery, enterprise SSD demand",
        "Memory oversupply returns before cash flow improves",
        "Treat as cyclical; size smaller than platform names.",
    ),
    "PLTR": (
        "AI operating system for government and enterprise workflows",
        "AIP adoption supports revenue acceleration and high software margins.",
        "Commercial customer growth, government renewals",
        "AIP pilots fail to convert into durable expansion",
        "Research candidate after revenue acceleration persists for two quarters.",
    ),
    "AMD": (
        "Second-source AI accelerator and CPU platform",
        "AI GPU ramp can add a new growth leg while server CPU share remains relevant.",
        "MI-series traction, server CPU share, margin expansion",
        "AI accelerator share stays niche and margins dilute",
        "Use pullbacks; watch execution versus NVIDIA ecosystem.",
    ),
    "AVGO": (
        "AI networking, custom silicon, and infrastructure software",
        "Custom ASIC and networking demand create AI infrastructure exposure with cash flow support.",
        "Custom silicon wins, VMware cash extraction",
        "Integration drag or AI order lumpiness",
        "High-quality compounder; valuation discipline required.",
    ),
    "TSM": (
        "Advanced-node manufacturing bottleneck",
        "AI and high-performance compute customers depend on advanced foundry capacity.",
        "N2 ramp, advanced packaging, capex discipline",
        "Geopolitical risk or pricing power erosion",
        "Core supply-chain watch; pair with geopolitical risk notes.",
    ),
    "ARM": (
        "Power-efficient compute IP across mobile, cloud, and edge AI",
        "Royalty model can participate in broader chip volume without fab risk.",
        "Data-center Arm adoption, royalty rate expansion",
        "AI value accrues to customers rather than IP holder",
        "Watch valuation; buy only when growth proof catches up.",
    ),
    "SOUN": (
        "Voice AI interface layer for physical-world workflows",
        "Restaurant and automotive deployments offer operating leverage if retention holds.",
        "Enterprise wins, recurring revenue growth",
        "Customer concentration or cash burn worsens",
        "Speculative satellite only; require proof of durable unit economics.",
    ),
    "CRWV": (
        "GPU cloud capacity for AI model builders",
        "Demand for specialized AI cloud can grow quickly if utilization and funding remain strong.",
        "Utilization, customer concentration, financing terms",
        "Debt burden or GPU supply normalization pressures returns",
        "Track as high-beta infrastructure name, not core holding yet.",
    ),
}


def load_tickers():
    try:
        import yaml

        with CONFIG_PATH.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)["watchlist"]
    except ImportError:
        return [
            {"ticker": "NVDA", "company": "NVIDIA", "tags": ["AI", "semiconductors", "data center"]},
            {"ticker": "TSLA", "company": "Tesla", "tags": ["EV", "autonomy", "energy storage"]},
            {"ticker": "SNDK", "company": "Sandisk", "tags": ["memory", "storage", "AI infrastructure"]},
            {"ticker": "PLTR", "company": "Palantir", "tags": ["AI software", "government", "enterprise"]},
            {"ticker": "AMD", "company": "Advanced Micro Devices", "tags": ["semiconductors", "AI accelerators", "CPU"]},
            {"ticker": "AVGO", "company": "Broadcom", "tags": ["semiconductors", "networking", "infrastructure software"]},
            {"ticker": "TSM", "company": "Taiwan Semiconductor Manufacturing", "tags": ["foundry", "advanced nodes", "AI supply chain"]},
            {"ticker": "ARM", "company": "Arm Holdings", "tags": ["chip IP", "mobile", "edge AI"]},
            {"ticker": "SOUN", "company": "SoundHound AI", "tags": ["voice AI", "automotive", "restaurants"]},
            {"ticker": "CRWV", "company": "CoreWeave", "tags": ["AI cloud", "GPUs", "data center"]},
        ]


def make_stock_prices(watchlist):
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2025-01-02", periods=260)
    rows = []
    for item in watchlist:
        ticker = item["ticker"]
        start, drift, vol, _ = PROFILES[ticker]
        price = start
        for i, date in enumerate(dates):
            shock = rng.normal(drift, vol)
            if i in (75, 150, 215) and ticker in {"NVDA", "PLTR", "AVGO", "CRWV"}:
                shock += 0.06
            if i in (95, 185) and ticker in {"TSLA", "SOUN", "SNDK"}:
                shock -= 0.045
            open_price = price * (1 + rng.normal(0, vol / 3))
            close = max(1, price * (1 + shock))
            high = max(open_price, close) * (1 + abs(rng.normal(0.006, vol / 6)))
            low = min(open_price, close) * (1 - abs(rng.normal(0.006, vol / 6)))
            volume = int((rng.lognormal(15, 0.35)) * (1 + abs(shock) * 8))
            rows.append([date, ticker, open_price, high, low, close, volume])
            price = close
    return pd.DataFrame(rows, columns=["date", "ticker", "open", "high", "low", "close", "volume"])


def make_crypto_prices():
    rng = np.random.default_rng(7)
    hours = pd.date_range("2026-05-01", periods=24 * 45, freq="h")
    rows = []
    for symbol, start, vol in [("BTC-USD", 105000, 0.012), ("ETH-USD", 5200, 0.017)]:
        price = start
        for i, ts in enumerate(hours):
            cycle = np.sin(i / 24 / 3) * 0.003
            shock = rng.normal(cycle, vol)
            open_price = price
            close = max(100, price * (1 + shock))
            high = max(open_price, close) * (1 + abs(rng.normal(0.003, vol / 5)))
            low = min(open_price, close) * (1 - abs(rng.normal(0.003, vol / 5)))
            volume = int(rng.lognormal(12, 0.45))
            rows.append([ts, symbol, open_price, high, low, close, volume])
            price = close
    return pd.DataFrame(rows, columns=["datetime", "symbol", "open", "high", "low", "close", "volume"])


def make_financials(watchlist):
    quarters = pd.date_range("2024-03-31", periods=9, freq="QE")
    rows = []
    for item in watchlist:
        ticker = item["ticker"]
        _, _, _, quality = PROFILES[ticker]
        for i, quarter in enumerate(quarters):
            growth = min(0.95, 0.08 + quality * 0.045 + i * 0.014 + (0.035 if ticker in {"NVDA", "PLTR", "CRWV"} else 0))
            gross_margin = min(0.82, 0.32 + quality * 0.06 + i * 0.004)
            ocf_margin = min(0.48, 0.04 + quality * 0.035 + i * 0.005)
            net_margin = min(0.38, 0.02 + quality * 0.03 + i * 0.004)
            guide_change = "raised" if i >= 6 and quality >= 4 else ("stable" if i >= 4 else "mixed")
            rows.append([quarter, ticker, growth, gross_margin, ocf_margin, net_margin, guide_change])
    return pd.DataFrame(
        rows,
        columns=[
            "quarter",
            "ticker",
            "revenue_growth_yoy",
            "gross_margin",
            "operating_cash_flow_margin",
            "net_income_margin",
            "guide_change",
        ],
    )


def make_notes(watchlist):
    rows = []
    for item in watchlist:
        ticker = item["ticker"]
        _, _, _, quality = PROFILES[ticker]
        narrative, evidence, catalysts, invalidation, buy_plan = NOTES[ticker]
        scores = [
            min(5, round(quality)),
            min(5, round(quality - 0.2)),
            min(5, round(quality - 0.3)),
            min(5, round(quality - 0.4)),
            min(5, round(quality)),
            4 if ticker in {"TSLA", "SNDK", "SOUN", "CRWV"} else 3,
            2 if ticker in {"NVDA", "AVGO", "ARM"} else 3,
        ]
        rows.append(
            [
                ticker,
                item["company"],
                "; ".join(item["tags"]),
                narrative,
                evidence,
                catalysts,
                invalidation,
                buy_plan,
                *scores,
            ]
        )
    return pd.DataFrame(
        rows,
        columns=[
            "ticker",
            "company",
            "tags",
            "narrative",
            "growth_evidence",
            "catalysts",
            "invalidation",
            "buy_plan",
            "narrative_space",
            "revenue_acceleration",
            "margin_quality",
            "cash_flow",
            "value_chain_position",
            "market_disagreement",
            "valuation_tolerance",
        ],
    )


def make_kronos_sample(stock_prices):
    rows = []
    for ticker, group in stock_prices.groupby("ticker"):
        last = group.sort_values("date").tail(1).iloc[0]
        base = last["close"]
        for step, date in enumerate(pd.bdate_range(last["date"] + pd.Timedelta(days=1), periods=10), 1):
            predicted = base * (1 + 0.004 * step + np.sin(step / 2) * 0.006)
            rows.append([date, ticker, predicted, predicted * 0.96, predicted * 1.04, "research_only_stub"])
    return pd.DataFrame(rows, columns=["date", "ticker", "predicted_close", "lower_band", "upper_band", "source"])


def make_paper_trades():
    now = pd.Timestamp("2026-06-30 16:00:00")
    return pd.DataFrame(
        [
            [now - pd.Timedelta(days=12), now - pd.Timedelta(days=8), "NVDA", "us_stock_daily", 0.12, 0.034, "closed"],
            [now - pd.Timedelta(days=7), now - pd.Timedelta(days=2), "PLTR", "us_stock_daily", 0.10, -0.018, "closed"],
            [now - pd.Timedelta(hours=54), now - pd.Timedelta(hours=20), "BTC-USD", "crypto_hourly", 0.08, 0.022, "closed"],
            [now - pd.Timedelta(hours=14), pd.NaT, "ETH-USD", "crypto_hourly", 0.06, 0.009, "open"],
        ],
        columns=["entry_time", "exit_time", "symbol", "strategy", "position_pct", "return_pct", "status"],
    )


def main():
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    watchlist = load_tickers()
    stock_prices = make_stock_prices(watchlist)
    stock_prices.to_csv(SAMPLE_DIR / "us_stock_ohlcv.csv", index=False)
    make_crypto_prices().to_csv(SAMPLE_DIR / "crypto_ohlcv_hourly.csv", index=False)
    make_financials(watchlist).to_csv(SAMPLE_DIR / "financial_metrics.csv", index=False)
    make_notes(watchlist).to_csv(SAMPLE_DIR / "watchlist_notes.csv", index=False)
    make_kronos_sample(stock_prices).to_csv(SAMPLE_DIR / "kronos_forecast_sample.csv", index=False)
    make_paper_trades().to_csv(SAMPLE_DIR / "paper_trades_sample.csv", index=False)
    print(f"Wrote sample data to {SAMPLE_DIR}")


if __name__ == "__main__":
    main()
