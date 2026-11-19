"""Chat settings tab."""

import streamlit as st

from ....rag_application import RAGApplication
from ...i18n import t


def render_chat_tab(app: RAGApplication):
    """Chat settings tab."""
    st.markdown(f"##### 🧠 {t('tab_chat')}")
    use_chat_memory = st.toggle(
        "Include recent conversation when answering",
        value=app.config.config.get("use_chat_memory", True),
        help="Lets follow-up questions like \"what about its capital?\" be understood "
             "in context. The recent turns are only used to interpret the question — "
             "the answer itself is still grounded only in the retrieved documents.",
    )
    memory_turns = st.number_input(
        "Number of recent exchanges to remember",
        min_value=1, max_value=10,
        value=int(app.config.config.get("memory_turns", 3)),
        disabled=not use_chat_memory,
    )

    st.markdown("---")
    st.markdown("##### ⚡ Answer cache")
    use_answer_cache = st.toggle(
        "Cache answers and reuse them instantly for repeated questions",
        value=app.config.config.get("use_answer_cache", True),
        help="A question you've asked before is answered instantly from cache — no "
             "retrieval, no LLM call — as long as your documents, document selection "
             "and retrieval settings haven't changed since. Automatically invalidated "
             "when you add, remove, or clear documents, or change retrieval or LLM "
             "backend settings.",
    )
    cache_count = app.persistent_store.answer_cache_count()
    st.caption(f"{cache_count} cached answer(s) stored.")
    if st.button("🗑️ Clear answer cache"):
        app.persistent_store.clear_answer_cache()
        st.success("Answer cache cleared.")
        st.rerun()

    if st.button("💾 Save Chat Settings", type="primary", use_container_width=True):
        app.config.config["use_chat_memory"] = use_chat_memory
        app.config.config["memory_turns"] = memory_turns
        app.config.config["use_answer_cache"] = use_answer_cache
        app.config.save_config()
        st.success("Settings saved.")
