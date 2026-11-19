"""Small reusable UI building blocks (hero, empty state, chat bubble, chips...)."""

import html as _html
import re
import streamlit as st
from typing import List

from ..dependencies import IMPORT_ERRORS
from .i18n import current_lang, detect_text_direction, is_rtl_lang, t


def _strip_leading_symbols(text: str) -> str:
    """'💬 Chat with Your Documents' -> 'Chat with Your Documents'.
    The header relies on type, not emoji, to carry its tone."""
    return re.sub(r"^[^\w]+", "", text, flags=re.UNICODE).strip() or text


def hero(title: str, subtitle: str):
    rtl = is_rtl_lang(current_lang())
    st.markdown(
        f'<div class="page-head" dir="{"rtl" if rtl else "ltr"}">'
        f"<h1>{_html.escape(_strip_leading_symbols(title))}</h1>"
        f"<p>{_html.escape(subtitle)}</p></div>",
        unsafe_allow_html=True,
    )


def empty_state(emoji: str, title: str, subtitle: str):
    rtl = is_rtl_lang(current_lang())
    st.markdown(
        f'<div class="empty-state" dir="{"rtl" if rtl else "ltr"}">'
        f'<div class="emoji">{emoji}</div>'
        f'<div class="es-title">{_html.escape(title)}</div>'
        f'<div class="es-sub">{_html.escape(subtitle)}</div></div>',
        unsafe_allow_html=True,
    )


def bidi_wrap(text: str, cursor: bool = False) -> str:
    """Wrap text in a per-message direction container AND keep Markdown
    working inside it (the blank lines are what make that work).
    '<' is escaped so model output / document text can never inject HTML."""
    direction = detect_text_direction(text)
    align = "right" if direction == "rtl" else "left"
    safe = (text or "").replace("<", "&lt;")
    if cursor:
        safe += " ▌"
    return f'<div dir="{direction}" style="text-align:{align};">\n\n{safe}\n\n</div>'


def render_chat_message(role: str, content: str, avatar: str):
    """Render one chat bubble with per-message direction detection, so a
    Persian message and an English message in the same thread each get
    correct alignment instead of the whole chat being forced one way."""
    with st.chat_message(role, avatar=avatar):
        st.markdown(bidi_wrap(content), unsafe_allow_html=True)


def doc_row_html(icon: str, name: str, size_text: str) -> str:
    """One-card document row for the Library tab."""
    return (
        f'<div class="doc-row"><span style="font-size:1.25rem;">{icon}</span>'
        f'<span class="doc-name">{_html.escape(name)}</span>'
        f'<span class="doc-size">{_html.escape(size_text)}</span></div>'
    )


def brand_html() -> str:
    return (
        '<div class="brand"><div class="brand-mark">📚</div>'
        f'<span class="brand-name">{_html.escape(t("app_title"))}</span></div>'
    )


def show_import_errors_banner():
    """Show any dependency import failures as a real, readable error instead
    of letting the script crash silently into a blank page. Kept in English
    /LTR regardless of UI language since it surfaces raw package/exception
    names that are not translatable."""
    if not IMPORT_ERRORS:
        return
    critical = [e for e in IMPORT_ERRORS if e[0] in (
        "RecursiveCharacterTextSplitter", "HuggingFaceEmbeddings", "Chroma",
    )]
    st.error(
        f"⚠️ {len(IMPORT_ERRORS)} optional package(s) failed to import. "
        f"Missing file-type support is skipped automatically; anything listed "
        f"under **Core (required)** below will stop the app from working."
    )
    with st.expander("Show import details", expanded=bool(critical)):
        if critical:
            st.markdown("**Core (required):**")
            for label, hint, err in critical:
                st.code(f"{label}\n  {err}\n  fix: {hint}", language="text")
        others = [e for e in IMPORT_ERRORS if e not in critical]
        if others:
            st.markdown("**Optional (file-type support only):**")
            for label, hint, err in others:
                st.code(f"{label}\n  {err}\n  fix: {hint}", language="text")


def api_status_label(app) -> str:
    """Short, translated label for the OpenAI-compatible backend status."""
    code, _detail = app.llm_manager.api_status()
    labels = {
        "not_used": "⚪ " + t("api_not_used"),
        "no_key": "🔴 " + t("api_no_key"),
        "blocked": "🔒 " + t("api_blocked"),
        "not_ready": "🟠 " + t("api_not_ready"),
        "ready": "🟡 " + t("api_ready"),
        "connected": "🟢 " + t("api_connected"),
        "failed": "🔴 " + t("api_failed"),
    }
    return labels.get(code, code)


def chips(items: List[tuple]) -> str:
    return "".join(f'<span class="chip {kind}">{_html.escape(str(text))}</span>' for text, kind in items)


def safe_html(text: str) -> str:
    """Escape for embedding in an HTML block inside markdown (no blank lines, no $ math)."""
    esc_ = _html.escape(re.sub(r"\n{2,}", "\n", (text or "").strip()))
    return esc_.replace("$", "&#36;").replace("\n", "<br>")


def short(text: str, n: int = 30) -> str:
    return text if len(text) <= n else text[:n - 1] + "…"
