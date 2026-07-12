from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXPORT_DIR = ROOT / "data" / "exports"


def export_markdown_report(name, markdown):
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)
    path = EXPORT_DIR / f"{safe}.md"
    path.write_text(markdown, encoding="utf-8")
    return path


def export_text_report(name, markdown):
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)
    path = EXPORT_DIR / f"{safe}.txt"
    path.write_text(markdown, encoding="utf-8")
    return path


def report_export_capabilities():
    return [
        {"format": "Markdown", "status": "ready", "note": "No extra dependency required."},
        {"format": "Word", "status": "planned", "note": "Can be enabled with python-docx."},
        {"format": "PDF", "status": "planned", "note": "Can be enabled with reportlab or browser print."},
    ]
