def format_pct(value):
    return f"{value * 100:.0f}%"


def build_weekly_report(scored_watchlist, bot_summaries, volume_price_note=None):
    top = scored_watchlist.head(5)
    lines = [
        "# Weekly Trade Lab Review",
        "",
        "## Long-term observation desk",
        f"- Watchlist size: {len(scored_watchlist)}",
        f"- Deep research candidates: {(scored_watchlist['bucket'] == 'Deep research candidate').sum()}",
        "- Top candidates: "
        + ", ".join(f"{row.ticker} ({row.total_score}/35)" for row in top.itertuples()),
        "",
        "## Paper bot review",
    ]
    for name, summary in bot_summaries.items():
        recommendation = "continue"
        if summary["status"] == "STOP":
            recommendation = "pause"
        elif summary["metrics"]["max_drawdown"] < -0.08:
            recommendation = "review"
        lines.append(
            f"- {name}: {recommendation}; return {format_pct(summary['metrics']['total_return'])}, "
            f"max drawdown {format_pct(summary['metrics']['max_drawdown'])}, "
            f"trades {summary['metrics']['trade_count']}, status {summary['status']}."
        )
    if volume_price_note:
        lines.extend(
            [
                "",
                "## Volume-price note",
                f"- {volume_price_note}",
            ]
        )
    lines.extend(
        [
            "",
            "## Operating note",
            "Kronos-style forecasts and AI summaries are research context only. No real orders are placed in V0.1.",
        ]
    )
    return "\n".join(lines)
