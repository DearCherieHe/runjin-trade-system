from datetime import datetime

import pandas as pd

from src.core.cache import CacheManager, stable_key
from src.data_sources.live_sources import LiveSourceError, fetch_yfinance_prices


MARKET_QUOTES_COLUMNS = [
    "synced_at",
    "symbol",
    "name",
    "price",
    "change_pct",
    "volume",
    "amount",
    "source",
]


def _empty_quote(symbol, source, reason):
    return {
        "symbol": symbol,
        "name": "",
        "price": None,
        "change_pct": None,
        "volume": None,
        "amount": None,
        "source": source,
        "reason": reason,
    }


def _normalize_akshare_row(symbol, frame, source):
    if frame is None or frame.empty:
        raise LiveSourceError(f"{source} returned no rows")
    row = frame.iloc[0]
    return {
        "symbol": str(row.get("代码", row.get("symbol", symbol))),
        "name": str(row.get("名称", row.get("name", ""))),
        "price": row.get("最新价", row.get("收盘", row.get("close"))),
        "change_pct": row.get("涨跌幅", row.get("change_pct")),
        "volume": row.get("成交量", row.get("volume")),
        "amount": row.get("成交额", row.get("amount")),
        "source": source,
        "reason": "",
    }


def fetch_akshare_quote_chain(symbol):
    try:
        import akshare as ak
    except ImportError as exc:
        raise LiveSourceError("akshare is not installed") from exc

    attempts = []
    providers = [
        ("stock_bid_ask_em", lambda: ak.stock_bid_ask_em(symbol=symbol)),
        ("stock_zh_a_spot", lambda: ak.stock_zh_a_spot()),
        ("stock_zh_a_spot_em", lambda: ak.stock_zh_a_spot_em()),
        ("stock_zh_a_hist", lambda: ak.stock_zh_a_hist(symbol=symbol, period="daily", adjust="qfq").tail(1)),
    ]
    for name, loader in providers:
        try:
            frame = loader()
            if name in {"stock_zh_a_spot", "stock_zh_a_spot_em"} and "代码" in frame.columns:
                frame = frame.loc[frame["代码"].astype(str) == str(symbol)]
            return _normalize_akshare_row(symbol, frame, name), attempts
        except Exception as exc:
            attempts.append({"provider": name, "status": "failed", "reason": str(exc)})
    raise LiveSourceError("AKShare quote chain failed: " + " | ".join(item["reason"] for item in attempts))


def fetch_yfinance_quote(symbol):
    frame = fetch_yfinance_prices([symbol], period="5d", interval="1d")
    latest = frame.sort_values("date").tail(1).iloc[0]
    previous = frame.sort_values("date").tail(2).head(1)
    change_pct = None
    if not previous.empty and previous.iloc[0]["close"]:
        change_pct = latest["close"] / previous.iloc[0]["close"] - 1
    return {
        "symbol": symbol,
        "name": "",
        "price": latest["close"],
        "change_pct": change_pct,
        "volume": latest["volume"],
        "amount": None,
        "source": "yfinance",
        "reason": "",
    }


def sync_single_quote(symbol, market="US", cache_ttl_seconds=120):
    cache = CacheManager()
    key = stable_key("quote", {"symbol": symbol, "market": market})
    cached = cache.get(key, ttl_seconds=cache_ttl_seconds)
    if cached:
        cached["cache"] = "hit"
        return cached

    attempts = []
    quote = None
    main_route = "akshare" if market in {"A_SHARE", "CN", "SH", "SZ"} else "yfinance"
    fallback_route = "yfinance" if main_route == "akshare" else "akshare"
    try:
        if main_route == "akshare":
            quote, chain_attempts = fetch_akshare_quote_chain(symbol)
            attempts.extend(chain_attempts)
        else:
            quote = fetch_yfinance_quote(symbol)
    except Exception as exc:
        attempts.append({"provider": main_route, "status": "failed", "reason": str(exc)})
        try:
            if fallback_route == "akshare":
                quote, chain_attempts = fetch_akshare_quote_chain(symbol)
                attempts.extend(chain_attempts)
            else:
                quote = fetch_yfinance_quote(symbol)
        except Exception as fallback_exc:
            attempts.append({"provider": fallback_route, "status": "failed", "reason": str(fallback_exc)})
            quote = _empty_quote(symbol, "none", str(fallback_exc))

    synced_at = datetime.now().isoformat(timespec="seconds")
    market_quotes_row = {
        "synced_at": synced_at,
        "symbol": quote["symbol"],
        "name": quote.get("name", ""),
        "price": quote.get("price"),
        "change_pct": quote.get("change_pct"),
        "volume": quote.get("volume"),
        "amount": quote.get("amount"),
        "source": quote.get("source"),
    }
    result = {
        "symbol": symbol,
        "market": market,
        "cache": "miss",
        "main_route": main_route,
        "fallback_route": fallback_route,
        "status": "success" if quote.get("source") != "none" else "failed",
        "failure_reason": quote.get("reason", ""),
        "attempts": attempts,
        "quote": quote,
        "market_quotes_saved": True,
        "market_quotes_row": market_quotes_row,
    }
    cache.set(key, result)
    return result


def result_to_frames(result):
    attempts = pd.DataFrame(result.get("attempts", []))
    if attempts.empty:
        attempts = pd.DataFrame([{"provider": result.get("main_route"), "status": result.get("status"), "reason": result.get("failure_reason", "")}])
    quotes = pd.DataFrame([result.get("market_quotes_row", {})], columns=MARKET_QUOTES_COLUMNS)
    summary = pd.DataFrame(
        [
            {
                "symbol": result.get("symbol"),
                "market": result.get("market"),
                "status": result.get("status"),
                "main_route": result.get("main_route"),
                "fallback_route": result.get("fallback_route"),
                "failure_reason": result.get("failure_reason"),
                "market_quotes_saved": result.get("market_quotes_saved"),
                "cache": result.get("cache"),
            }
        ]
    )
    return summary, attempts, quotes
