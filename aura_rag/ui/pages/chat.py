"""Chat page."""

import logging
import streamlit as st
from datetime import datetime
from typing import List, Optional

from ...constants import NO_ANSWER_MESSAGES
from ...rag_application import RAGApplication
from ..chat_export import chat_history_to_markdown
from ..chat_render import render_answer_extras, render_saved_message
from ..components import bidi_wrap, empty_state, hero, render_chat_message, short
from ..i18n import current_lang, t

logger = logging.getLogger(__name__)


def ensure_conversation(app):
    store = app.persistent_store
    ids = {c["id"] for c in store.list_conversations()}
    if st.session_state.get("conv_id") not in ids:
        convs = store.list_conversations()
        st.session_state.conv_id = convs[0]["id"] if convs else store.new_conversation()


def render_conversation_sidebar(app):
    """Chat list in the sidebar: new, switch, rename, export, delete."""
    store = app.persistent_store
    convs = store.list_conversations()
    if st.button("＋  " + t("new_chat"), use_container_width=True, key="new_chat_btn"):
        latest = convs[0] if convs else None
        st.session_state.conv_id = latest["id"] if latest and latest["n"] == 0 else store.new_conversation()
        st.rerun()
    # Empty chats aren't listed: the "New chat" button above already represents them,
    # so a blank chat never shows up as a second "New chat".
    recent = [c for c in convs if c["n"] > 0][:12]
    if recent:
        st.caption(t("recent_chats"))
    for c in recent:
        active = c["id"] == st.session_state.conv_id
        key = f"convactive_{c['id']}" if active else f"conv_{c['id']}"
        if st.button(short(c["title"]), key=key, use_container_width=True,
                     help=f"{c['n']} messages"):
            st.session_state.conv_id = c["id"]
            st.rerun()
    current = next((c for c in convs if c["id"] == st.session_state.conv_id), None)
    if current and current["n"] > 0:
        with st.expander(t("manage_chat")):
            new_title = st.text_input(t("rename_chat"), current["title"],
                                      key=f"title_{current['id']}_{abs(hash(current['title']))}")
            if new_title.strip() and new_title != current["title"]:
                store.rename_conversation(current["id"], new_title)
                st.rerun()
            msgs = store.load_messages(current["id"])
            st.download_button(t("export_chat"), chat_history_to_markdown(msgs),
                               file_name="chat_export.md", mime="text/markdown",
                               use_container_width=True, disabled=not msgs, key="sidebar_export_btn")
            if st.button(t("delete_chat"), use_container_width=True, key="delete_chat_btn"):
                store.delete_conversation(current["id"])
                st.rerun()


def render_scope_selector(app: RAGApplication) -> Optional[List[str]]:
    """Let the user choose between answering from ALL documents or only from
    a SELECTED subset. Returns None for 'all documents', otherwise the list
    of selected file names (possibly empty). The choice is saved in
    config.json so it survives restarts."""
    cfg = app.config.config
    available = app.persistent_store.list_file_names()

    mode = cfg.get("doc_scope_mode", "all")
    if mode not in ("all", "selected"):
        mode = "all"
    # Drop files that no longer exist in the index (e.g. deleted documents).
    saved = [f for f in cfg.get("doc_scope_files", []) if f in available]

    if mode == "selected":
        summary = t("scope_active_selected").format(m=len(saved), n=len(available))
    else:
        summary = t("scope_active_all").format(n=len(available))

    with st.expander(f"📂 {t('scope_title')} — {summary}", expanded=False):
        options = {"all": t("scope_all"), "selected": t("scope_selected")}
        new_mode = st.radio(
            t("scope_title"),
            options=list(options.keys()),
            format_func=lambda k: options[k],
            index=list(options.keys()).index(mode),
            horizontal=True,
            label_visibility="collapsed",
            key="doc_scope_mode_radio",
        )
        new_files = saved
        if new_mode == "selected":
            new_files = st.multiselect(
                t("scope_pick"),
                options=available,
                default=saved,
                key="doc_scope_files_multiselect",
            )

        if new_mode != mode or (new_mode == "selected" and new_files != cfg.get("doc_scope_files", [])):
            cfg["doc_scope_mode"] = new_mode
            cfg["doc_scope_files"] = list(new_files)
            app.config.save_config()
            st.rerun()

    if mode == "selected":
        return list(saved)
    return None


def show_chat_page(app: RAGApplication):
    hero(t("chat_hero_title"), t("chat_hero_sub"))
    lang = current_lang()
    store = app.persistent_store
    conv_id = st.session_state.conv_id

    if app.vector_manager.get_document_count() == 0:
        empty_state("📄", t("no_docs_title"), t("no_docs_sub"))
        return

    llm_ready = app.llm_manager.is_ready()
    st.session_state.llm_loaded = llm_ready
    if not llm_ready:
        c1, c2 = st.columns([4, 1])
        c1.info(t("retrieval_only_notice"))
        if c2.button(t("load_default_model"), type="primary", use_container_width=True):
            with st.spinner("Loading LLM..."):
                if app.initialize_llm():
                    st.session_state.llm_loaded = True
                    st.rerun()
                else:
                    st.error(t("llm_load_failed"))

    st.markdown(
        f"<span class='app-pill accent'>🔍 {app.retriever.describe_active_pipeline()}</span>",
        unsafe_allow_html=True,
    )
    st.write("")

    # Which documents may be used to answer: None = all, list = only those.
    scope_files = render_scope_selector(app)
    scope_empty = scope_files is not None and len(scope_files) == 0
    if scope_empty:
        st.warning(t("scope_none_selected"))

    prior = store.load_messages(conv_id)
    typed = st.chat_input(t("chat_placeholder"), disabled=scope_empty)
    prompt = st.session_state.pop("_queued_q", None) or typed

    for message in prior:
        render_saved_message(app, message)

    if not prior and not prompt:
        empty_state("💡", t("ask_first_title"), t("ask_first_sub"))
        st.write("")
        cols = st.columns(2)
        for i in range(4):
            label = t(f"sugg_{i + 1}")
            if cols[i % 2].button(label, key=f"sugg_{i}", use_container_width=True, disabled=scope_empty):
                st.session_state["_queued_q"] = label
                st.rerun()

    if prompt:
        store.add_message(conv_id, "user", prompt)
        if not prior:
            store.rename_conversation(conv_id, prompt[:60])
        render_chat_message("user", prompt, "🧑")

        with st.chat_message("assistant", avatar="📚"):
            placeholder = st.empty()
            collected = []
            sources = []
            retrieval_time = None
            generation_time = None
            from_cache = False
            backend_used = None
            standalone = prompt
            error = None

            # Streamed token-by-token so the answer appears progressively. Prior turns (excluding the
            # just-saved current prompt) are passed along so follow-up questions resolve in context.
            with st.spinner(t("thinking")):
                for event in app.query_stream(
                    prompt, language=lang, chat_history=prior, file_names=scope_files
                ):
                    if event["type"] == "meta":
                        sources = event["sources"]
                        retrieval_time = event["retrieval_time"]
                        from_cache = event.get("from_cache", False)
                        standalone = event.get("standalone", prompt)
                    elif event["type"] == "token":
                        collected.append(event["text"])
                        placeholder.markdown(
                            bidi_wrap("".join(collected), cursor=True),
                            unsafe_allow_html=True,
                        )
                    elif event["type"] == "done":
                        generation_time = event["generation_time"]
                        backend_used = event.get("backend")
                    elif event["type"] == "error":
                        error = event["message"]

            if error:
                placeholder.empty()
                st.error(f"Error: {error}")
            else:
                answer = "".join(collected).strip()
                placeholder.markdown(bidi_wrap(answer), unsafe_allow_html=True)

                is_no_answer = answer in NO_ANSWER_MESSAGES.values()
                if is_no_answer:
                    sources = []
                audit = []
                if sources and backend_used != "retrieval":
                    with st.spinner(t("verifying")):
                        try:
                            audit = app.verify_answer(answer, sources)
                        except Exception as exc:
                            logger.exception("Answer verification failed")
                            st.caption(f"Verification skipped: {exc}")

                timing = {"retrieval": retrieval_time, "generation": generation_time,
                          "cached": from_cache, "backend": backend_used,
                          "standalone": standalone if standalone != prompt else None}
                debug = {} if from_cache else dict(app.retriever.last_debug)
                render_answer_extras(app, answer, sources, audit, debug, timing)

                store.add_message(conv_id, "assistant", answer,
                                  {"sources": sources, "audit": audit, "debug": debug, "timing": timing})

    st.write("")
    messages = store.load_messages(conv_id)
    if messages:
        c1, c2 = st.columns(2)
        with c1:
            if st.button(t("clear_chat"), use_container_width=True):
                store.clear_conversation(conv_id)
                st.rerun()
        with c2:
            st.download_button(
                t("export_chat"),
                data=chat_history_to_markdown(messages),
                file_name=f"chat_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
                use_container_width=True,
            )
