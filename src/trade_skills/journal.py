from pathlib import Path

import pandas as pd


JOURNAL_DIR = Path(__file__).resolve().parents[2] / "data" / "journal"


def default_deep_dive_sections(symbol: str, profile: dict | None = None) -> list[dict]:
    profile = profile or {}
    return [
        {"section": "Business identity", "prompt": "这家公司到底卖什么，客户是谁，为什么现在重要？", "current_note": profile.get("narrative", "")},
        {"section": "Fundamentals", "prompt": "收入、毛利、现金流是否已经兑现叙事？", "current_note": profile.get("growth_evidence", "")},
        {"section": "Technicals", "prompt": "趋势、相对强弱、关键均线和成交量是否支持继续观察？", "current_note": ""},
        {"section": "Catalysts", "prompt": "未来 3-12 个月有什么事件可能改变市场认知？", "current_note": profile.get("catalysts", "")},
        {"section": "Supply chain and peers", "prompt": "它在价值链中卡住了什么位置，替代品是谁？", "current_note": ""},
        {"section": "Narrative audit", "prompt": "哪些事实会证明我错了？市场共识可能在哪里过热？", "current_note": profile.get("invalidation", "")},
    ]


def list_journal_entries() -> pd.DataFrame:
    if not JOURNAL_DIR.exists():
        return pd.DataFrame(columns=["file", "symbol", "title", "updated"])
    rows = []
    for path in sorted(JOURNAL_DIR.glob("*.md")):
        name = path.stem
        symbol = name.split("_", 1)[0].upper() if "_" in name else "GENERAL"
        rows.append({"file": str(path), "symbol": symbol, "title": name.replace("_", " "), "updated": path.stat().st_mtime})
    return pd.DataFrame(rows)


def build_markdown_note(symbol: str, sections: pd.DataFrame) -> str:
    lines = [f"# {symbol.upper()} Deep Dive", "", "> Research journal. Not a trading instruction.", ""]
    for _, row in sections.iterrows():
        lines.extend([f"## {row['section']}", "", str(row.get("current_note", "") or row.get("prompt", "")), ""])
    return "\n".join(lines).strip() + "\n"

