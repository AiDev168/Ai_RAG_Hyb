"""Settings page: one module per tab."""

import streamlit as st

from ....rag_application import RAGApplication
from ...components import hero
from ...i18n import t
from .appearance_tab import render_appearance_tab
from .chat_tab import render_chat_tab
from .embeddings_tab import render_embeddings_tab
from .language_tab import render_language_tab
from .llm_tab import render_llm_tab
from .retrieval_tab import render_retrieval_tab
from .storage_tab import render_storage_tab



def show_settings_page(app: RAGApplication):
    hero(t("settings_hero_title"), t("settings_hero_sub"))

    tab_llm, tab_retrieval, tab_embed, tab_chat, tab_appearance, tab_storage, tab_language = st.tabs(
        [t("tab_llm"), t("tab_retrieval"), t("tab_embed"), t("tab_chat"),
         t("tab_appearance"), t("tab_storage"), t("tab_language")]
    )

    with tab_llm:
        render_llm_tab(app)
    with tab_retrieval:
        render_retrieval_tab(app)
    with tab_embed:
        render_embeddings_tab(app)
    with tab_chat:
        render_chat_tab(app)
    with tab_appearance:
        render_appearance_tab(app)
    with tab_storage:
        render_storage_tab(app)
    with tab_language:
        render_language_tab(app)
