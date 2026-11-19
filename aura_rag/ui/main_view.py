"""Application shell: page setup, sidebar, navigation and page dispatch."""

import streamlit as st
import traceback

from ..config import Config
from ..network import apply_network_mode, is_offline_mode
from ..rag_application import RAGApplication
from .components import api_status_label, brand_html, show_import_errors_banner
from .i18n import t
from .pages.chat import ensure_conversation, render_conversation_sidebar, show_chat_page
from .pages.documents import show_documents_page
from .pages.settings import show_settings_page
from .pages.status import show_status_page
from .theme import inject_custom_css


def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="Local Document Q&A", page_icon="📚", layout="wide"
    )

    try:
        if "app" not in st.session_state:
            st.session_state.app = RAGApplication()
    except Exception:
        # Language may not be resolvable yet if RAGApplication itself failed,
        # so this failure path is always shown in English.
        inject_custom_css()
        show_import_errors_banner()
        st.error(
            "❌ The app failed to start. This is almost always a missing or "
            "mismatched dependency — see the full error below and the fix "
            "commands in the import banner above."
        )
        st.code(traceback.format_exc(), language="text")
        if is_offline_mode():
            st.info(
                "Offline mode is on, so missing models can't be downloaded. If the error above is about "
                "a missing embedding model, copy that model into the app's models/embeddings folder — "
                "or switch to Online mode and let HuggingFace download it."
            )
            if st.button("🌐 Switch to Online mode and retry"):
                apply_network_mode(False)
                try:
                    _cfg = Config()
                    _cfg.config["network_mode"] = "online"
                    _cfg.save_config()
                except Exception:
                    pass
                st.rerun()
        st.stop()

    if "language" not in st.session_state:
        st.session_state.language = st.session_state.app.config.config.get("language", "en")
    if "theme" not in st.session_state:
        st.session_state.theme = st.session_state.app.config.config.get("theme", "dark")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "llm_loaded" not in st.session_state:
        st.session_state.llm_loaded = False
    if "confirm_delete_all" not in st.session_state:
        st.session_state.confirm_delete_all = False

    inject_custom_css()
    show_import_errors_banner()

    app = st.session_state.app
    if app.vector_manager.embeddings is None:
        st.warning(
            "The embedding model is not ready. Open Settings to choose a local model, or switch to Online "
            "mode to download the configured model. "
            + (app.vector_manager.embedding_error or "")
        )
    ensure_conversation(app)

    with st.sidebar:
        st.markdown(brand_html(), unsafe_allow_html=True)
        st.caption(t("app_tagline"))

        net_options = {"offline": "🔒 " + t("net_offline"), "online": "🌐 " + t("net_online")}
        current_net = "offline" if is_offline_mode() else "online"
        chosen_net = st.radio(
            t("network_label"),
            options=list(net_options.keys()),
            format_func=lambda k: net_options[k],
            index=list(net_options.keys()).index(current_net),
            horizontal=True,
            key="network_mode_radio",
            help=t("network_help"),
        )
        if chosen_net != current_net:
            apply_network_mode(chosen_net == "offline")
            app.config.config["network_mode"] = chosen_net
            app.config.save_config()
            if app.vector_manager.embeddings is None:
                with st.spinner("Loading the embedding model..."):
                    if app.vector_manager._initialize_embeddings():
                        app.vector_manager.load_vectorstore()
            # Re-evaluate the API backend under the new mode (remote endpoints
            # are only allowed online), then refresh the "LLM loaded" status.
            app.llm_manager.openai_client = None
            app.llm_manager.api_verified = None
            if app.llm_manager._openai_enabled():
                app.llm_manager.initialize_openai()
            st.session_state.llm_loaded = app.llm_manager.is_ready()
            st.rerun()
        st.caption(("🔒 " + t("offline_badge")) if is_offline_mode() else ("🌐 " + t("online_badge")))

        lang_options = {"en": "🇬🇧 English", "fa": "🇮🇷 فارسی"}
        selected_lang = st.selectbox(
            t("language_label"),
            options=list(lang_options.keys()),
            format_func=lambda k: lang_options[k],
            index=list(lang_options.keys()).index(st.session_state.language),
            key="language_selector",
        )
        if selected_lang != st.session_state.language:
            st.session_state.language = selected_lang
            app.config.config["language"] = selected_lang
            app.config.save_config()
            st.rerun()

        st.markdown("---")

        page_keys = ["nav_chat", "nav_documents", "nav_settings", "nav_status"]
        page = st.radio(
            "Navigation",
            [t(k) for k in page_keys],
            label_visibility="collapsed",
            key="nav_radio",
        )
        if page == t("nav_chat"):
            st.markdown("---")
            render_conversation_sidebar(app)

        st.markdown("---")
        st.markdown(f"**{t('overview')}**")

        doc_count = app.vector_manager.get_document_count()
        st.metric(t("indexed_chunks"), doc_count)

        total_size = app.doc_manager.get_total_size()
        st.metric(t("documents_size"), app.doc_manager.format_size(total_size))

        llm_status = t("loaded") if st.session_state.llm_loaded else t("not_loaded")
        st.metric(t("llm_status"), llm_status)
        st.metric(t("api_status_label"), api_status_label(app))

    if page == t("nav_chat"):
        show_chat_page(app)
    elif page == t("nav_documents"):
        show_documents_page(app)
    elif page == t("nav_settings"):
        show_settings_page(app)
    elif page == t("nav_status"):
        show_status_page(app)
