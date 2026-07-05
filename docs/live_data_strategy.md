# Live Data Strategy

RunJin Trade System now supports three data modes:

- `Live auto`: try live/near-live sources first, then fall back to bundled sample data.
- `Sample only`: use offline demo data only.
- `Live strict`: raise an error if live sources fail.

## Currently Wired Sources

| Layer | Source | Cost | Status | Notes |
|---|---:|---:|---|---|
| US equities OHLCV | yfinance / Yahoo Finance | Free unofficial | Wired | Good for research dashboards; not low-latency execution. |
| Crypto OHLCV | Binance public REST klines | Free | Wired | Hourly BTC/ETH bars, no API key. |
| China market stack | finshare / OpenDataTools / TuShare | Free / mixed | Registered optional | Installed only when needed; useful for A-share/HK/China macro style coverage. |
| Broker API | Tiger OpenAPI | Account-gated | Placeholder | Credentials and package will be enabled later. |
| Fundamentals | Bundled sample data | Free | Fallback | Real fundamentals need paid APIs or SEC companyfacts integration. |
| Filings | SEC EDGAR APIs | Free official | Planned | Best legal early source for 8-K, 10-Q, 10-K, Form 4, S-1. |
| News | Not wired | Mixed | Planned | Free RSS is noisy; paid Benzinga/Polygon/Finnhub is more useful. |

## Paid Upgrade Priority

1. **Market data**
   - Polygon/Massive: US equities, options, crypto, snapshots, trades, quotes, news.
   - Alpaca Market Data: real-time/historical equities, options, crypto, WebSocket.
   - Finnhub: real-time WebSocket trades, company news, fundamentals, earnings calendar.

## K-line History Requirements

RunJin supports `1H / 1D / 1W / 1M / 1Q / 1Y` chart periods.

- `1D / 1W / 1M / 1Q / 1Y`: built from daily OHLCV. In live mode the US equity adapter requests `period=max` from yfinance, then resamples weekly/monthly/quarterly/yearly locally.
- `1H`: free sources usually limit historical coverage. The current adapter tries yfinance 1-hour data on demand and labels the coverage range. For true long-range historical hourly bars, use Polygon, Alpaca, Tiger OpenAPI, or another paid/broker data feed.
- `Replay as of`: every K-line page can treat a selected historical date as “now” and hides bars after that date, so user judgments can be tested without future leakage.

1.5. **Broker/account data**
   - Tiger OpenAPI: account-gated broker API for market data, account, and trading workflows. In RunJin V0.1 this is only a credential placeholder and must not place orders.

2. **News and catalyst feeds**
   - Benzinga Pro / Benzinga API for fast retail-accessible news.
   - Polygon News or Finnhub News for API-native integration.

3. **Fundamentals and estimates**
   - Financial Modeling Prep for practical API coverage.
   - Intrinio / FactSet if budget allows institutional-quality data.

4. **Alternative data**
   - App usage, web traffic, job postings, GitHub activity, social attention, supply-chain checks.
   - These need careful validation because noisy alternative data can create false confidence.

## Early-Information Principle

The system should only use legal public information:

- Real-time price/volume and order-book-derived signals.
- SEC filings and official company disclosures.
- Earnings calendars, transcripts, guidance changes, and analyst estimate revisions.
- Newswire/API feeds.
- Public web signals and alternative data that do not violate terms or confidentiality.

Do not use material non-public information. The edge should come from faster ingestion, cleaner synthesis, and disciplined reaction, not from illegal access.
