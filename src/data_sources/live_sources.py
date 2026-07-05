import json
import urllib.parse
import urllib.request
from datetime import datetime, timezone

import pandas as pd


class LiveSourceError(RuntimeError):
    pass


def fetch_yfinance_prices(tickers, period="1y", interval="1d"):
    try:
        import yfinance as yf
    except ImportError as exc:
        raise LiveSourceError("yfinance is not installed") from exc

    raw = yf.download(
        tickers,
        period=period,
        interval=interval,
        group_by="ticker",
        auto_adjust=False,
        progress=False,
        threads=True,
    )
    if raw.empty:
        raise LiveSourceError("yfinance returned no rows")

    frames = []
    for ticker in tickers:
        if isinstance(raw.columns, pd.MultiIndex):
            if ticker not in raw.columns.get_level_values(0):
                continue
            data = raw[ticker].copy()
        else:
            data = raw.copy()
        data = data.reset_index()
        date_col = "Date" if "Date" in data.columns else "Datetime"
        required = ["Open", "High", "Low", "Close", "Volume"]
        if not set(required).issubset(data.columns):
            continue
        data = data[[date_col, *required]].rename(
            columns={
                date_col: "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )
        data["ticker"] = ticker
        frames.append(data)

    if not frames:
        raise LiveSourceError("yfinance rows could not be normalized")
    prices = pd.concat(frames, ignore_index=True)
    prices["date"] = pd.to_datetime(prices["date"]).dt.tz_localize(None)
    return prices[["date", "ticker", "open", "high", "low", "close", "volume"]].dropna()


def fetch_binance_hourly(symbol_map, limit=1000):
    frames = []
    for app_symbol, exchange_symbol in symbol_map.items():
        query = urllib.parse.urlencode(
            {"symbol": exchange_symbol, "interval": "1h", "limit": min(limit, 1000)}
        )
        url = f"https://api.binance.com/api/v3/klines?{query}"
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "RunJinTradeSystem/0.1 research dashboard"},
        )
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            raise LiveSourceError(f"Binance request failed for {exchange_symbol}") from exc
        if not payload:
            raise LiveSourceError(f"Binance returned no rows for {exchange_symbol}")
        rows = []
        for row in payload:
            rows.append(
                [
                    datetime.fromtimestamp(row[0] / 1000, tz=timezone.utc).replace(tzinfo=None),
                    app_symbol,
                    float(row[1]),
                    float(row[2]),
                    float(row[3]),
                    float(row[4]),
                    float(row[5]),
                ]
            )
        frames.append(
            pd.DataFrame(rows, columns=["datetime", "symbol", "open", "high", "low", "close", "volume"])
        )
    if not frames:
        raise LiveSourceError("No Binance frames were built")
    return pd.concat(frames, ignore_index=True)


def fetch_finshare_snapshots(symbols):
    try:
        import finshare as fs
    except ImportError as exc:
        raise LiveSourceError("finshare is not installed") from exc
    try:
        snapshots = fs.get_batch_snapshots(symbols)
    except Exception as exc:
        raise LiveSourceError("finshare snapshot request failed") from exc
    rows = []
    if isinstance(snapshots, dict):
        for symbol, snap in snapshots.items():
            rows.append(
                {
                    "symbol": symbol,
                    "price": getattr(snap, "last_price", None),
                    "change_pct": getattr(snap, "change_pct", None),
                    "volume": getattr(snap, "volume", None),
                    "source": "finshare",
                }
            )
    elif isinstance(snapshots, pd.DataFrame):
        result = snapshots.copy()
        result["source"] = "finshare"
        return result
    if not rows:
        raise LiveSourceError("finshare returned no snapshot rows")
    return pd.DataFrame(rows)


def fetch_opendatatools_quotes(symbols):
    try:
        from opendatatools import stock
    except ImportError as exc:
        raise LiveSourceError("opendatatools is not installed") from exc
    df, message = stock.get_quote(",".join(symbols))
    if df is None or df.empty:
        raise LiveSourceError(f"opendatatools returned no rows: {message}")
    result = df.copy()
    result["source"] = "opendatatools"
    return result


def fetch_tushare_hist(symbol, start=None, end=None):
    try:
        import tushare as ts
    except ImportError as exc:
        raise LiveSourceError("tushare is not installed") from exc
    try:
        data = ts.get_hist_data(symbol, start=start, end=end)
    except Exception as exc:
        raise LiveSourceError("tushare historical request failed") from exc
    if data is None or data.empty:
        raise LiveSourceError("tushare returned no rows")
    result = data.reset_index().rename(
        columns={
            "date": "date",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
        }
    )
    result["symbol"] = symbol
    result["source"] = "tushare"
    return result


def tiger_openapi_status():
    try:
        import tigeropen  # noqa: F401
    except ImportError:
        return {
            "available": False,
            "source": "tiger_openapi",
            "message": "tigeropen package is not installed; credentials are not configured.",
        }
    return {
        "available": True,
        "source": "tiger_openapi",
        "message": "tigeropen package detected; waiting for credentials and account configuration.",
    }
