"""Chat transcript helpers."""

from datetime import datetime
from typing import List


def _format_source_label(source) -> str:
    """sources are now dicts ({file_name, page, snippet, ocr}); this renders
    a short label for contexts (exports, the sidebar) that just need the
    filename+page, not the full expander UI."""
    if isinstance(source, dict):
        label = source.get("file_name", "Unknown")
        pages = source.get("pages") or source.get("page", "")
        if pages != "":
            label += f" ({pages})"
        if source.get("ocr"):
            label += " [OCR]"
        return label
    return str(source)


def chat_history_to_markdown(history: List[dict]) -> str:
    """Serialize the chat transcript to a Markdown document, preserving
    per-message direction hints as HTML comments-free plain text so it
    still reads correctly when opened in any editor or viewer."""
    lines = ["# Chat Export", f"_Exported {datetime.now().strftime('%Y-%m-%d %H:%M')}_", ""]
    for msg in history:
        who = "**You**" if msg["role"] == "user" else "**Assistant**"
        lines.append(f"{who}:")
        lines.append("")
        lines.append(msg["content"])
        if msg.get("sources"):
            lines.append("")
            lines.append("Sources: " + ", ".join(_format_source_label(s) for s in msg["sources"]))
        lines.append("")
        lines.append("---")
        lines.append("")
    return "\n".join(lines)
