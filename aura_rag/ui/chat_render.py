"""Rendering of answers: sources, citation audit, retrieval debug, saved messages."""

import html as _html
import streamlit as st
from typing import List, Optional

from ..text_utils import cited_numbers
from .components import bidi_wrap, chips, render_chat_message, safe_html
from .i18n import detect_text_direction, t


AUDIT_KIND = {"ok": "ok", "uncited": "info", "miscited": "warn", "unsupported": "bad", "invalid": "bad"}
AUDIT_ICON = {"ok": "✅", "uncited": "📝", "miscited": "🔀", "unsupported": "⚠️", "invalid": "🚫"}


def render_meta(sources: List[dict], answer: str, timing: Optional[dict], audit: Optional[List[dict]]):
    cited = cited_numbers(answer)
    items = [(t("chip_sources").format(n=len(sources)), "muted"),
             (t("chip_cited").format(n=len([s for s in sources if s.get("n") in cited])), "info")]
    if audit:
        ratio = sum(a["status"] == "ok" for a in audit) / len(audit)
        items.append((t("chip_verified").format(p=f"{ratio:.0%}"), "ok" if ratio >= 0.8 else "warn" if ratio >= 0.5 else "bad"))
    st.markdown(chips(items), unsafe_allow_html=True)


def render_sources(sources: List[dict], answer: str = ""):
    """Source cards: numbered, relevance bar, 'cited' badge, expandable full passage."""
    if not sources:
        return
    cited = cited_numbers(answer)
    n_cited = len([s for s in sources if s.get("n") in cited])
    scores = [float(s.get("score", 0.0)) for s in sources]
    lo, hi = min(scores), max(scores)
    with st.expander(f"{t('sources')} · {len(sources)} · {t('chip_cited').format(n=n_cited)}"):
        for s in sources:
            is_cited = s.get("n") in cited
            where = s.get("file_name", "Unknown")
            if s.get("pages"):
                where += f" · {s['pages']}"
            if s.get("ocr"):
                where += " · OCR"
            pct = 100 if hi <= lo else max(6, int(100 * (float(s.get("score", 0.0)) - lo) / (hi - lo)))
            text = s.get("text") or s.get("snippet", "")
            direction = detect_text_direction(text)
            badge = chips([(t("src_cited"), "info")]) if is_cited else ""
            card_cls = "src-card cited" if is_cited else "src-card"
            st.markdown(
                f'<div class="{card_cls}"><div class="src-head"><span class="src-num">{s.get("n", "•")}</span>'
                f'<span class="src-title">{_html.escape(where)}</span>{badge}</div>'
                f'<div class="src-meta">{t("src_relevance")} {float(s.get("score", 0.0)):.2f}</div>'
                f'<div class="bar"><span style="width:{pct}%"></span></div>'
                f'<details><summary>{t("src_show")}</summary>'
                f'<div class="src-text" dir="{direction}">{safe_html(text)}</div></details></div>',
                unsafe_allow_html=True,
            )


_render_sources_expander = render_sources  # backward-compatible name


def render_audit(audit: List[dict]):
    if not audit:
        return
    counts: dict = {}
    for a in audit:
        counts[a["status"]] = counts.get(a["status"], 0) + 1
    with st.expander(t("evidence_check").format(ok=counts.get("ok", 0), total=len(audit))):
        st.markdown(chips([(f"{AUDIT_ICON[s]} {n} {t('audit_' + s)}", AUDIT_KIND[s])
                           for s, n in sorted(counts.items(), key=lambda x: -x[1])]), unsafe_allow_html=True)
        st.caption(t("evidence_note"))
        for a in audit:
            hint = f" · [{a['best_source']}]" if a["status"] in ("miscited", "uncited") else ""
            direction = detect_text_direction(a["claim"])
            st.markdown(
                f'<div class="aud {AUDIT_KIND[a["status"]]}">{AUDIT_ICON[a["status"]]} '
                f'<b>{_html.escape(t("audit_" + a["status"]))}</b> '
                f'<small>sim {a["support"]:.2f}{_html.escape(hint)}</small>'
                f'<br><span dir="{direction}">{safe_html(a["claim"])}</span></div>',
                unsafe_allow_html=True,
            )


def render_answer_extras(app, answer: str, sources: List[dict], audit: List[dict],
                         debug: Optional[dict], timing: Optional[dict]):
    """Everything shown under an answer: chips, source cards, evidence check, timings, debug."""
    timing = timing or {}
    if sources:
        render_meta(sources, answer, timing, audit)
        render_sources(sources, answer)
    render_audit(audit)
    if timing.get("cached"):
        st.caption(f"⚡ {t('cached_answer_label')}")
    elif timing.get("retrieval") is not None and timing.get("generation") is not None:
        st.caption(f"⏱️ {t('retrieval_time_label')}: {timing['retrieval']:.1f}s · "
                   f"{t('generation_time_label')}: {timing['generation']:.1f}s")
    if timing.get("standalone"):
        st.caption(f"🔎 {t('searched_as')}: {timing['standalone']}")
    if timing.get("backend") == "openai":
        ep = app.llm_manager.describe_openai_endpoint()
        if ep["local"]:
            st.caption(f"🖥️ {t('local_endpoint_notice')} ({ep['host']})")
        else:
            st.caption(f"☁️ {t('cloud_answer_notice')} ({ep['host']})")
    if debug and app.config.config.get("show_retrieval_debug", True):
        with st.expander("🛠️ Retrieval debug"):
            if timing.get("cached"):
                st.caption("This answer came from the answer cache (no retrieval ran). "
                           "Clear the cache in Settings → Chat & Cache to re-test.")
            else:
                st.json(debug)


def render_saved_message(app, m: dict):
    if m["role"] == "user":
        render_chat_message("user", m["content"], "🧑")
        return
    with st.chat_message("assistant", avatar="📚"):
        st.markdown(bidi_wrap(m["content"]), unsafe_allow_html=True)
        render_answer_extras(app, m["content"], m.get("sources") or [], m.get("audit") or [],
                             m.get("debug"), m.get("timing"))
