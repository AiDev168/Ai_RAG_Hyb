"""Storage settings tab."""

import streamlit as st

from ....rag_application import RAGApplication


def render_storage_tab(app: RAGApplication):
    """Storage settings tab."""
    for label, path in [
        ("Base Directory", app.config.base_dir),
        ("Documents", app.config.docs_dir),
        ("Vector DB", app.config.db_dir),
        ("Models", app.config.models_dir),
    ]:
        st.markdown(
            f"<div class='app-card' dir='ltr' style='text-align:left;'>"
            f"<div class='app-card-title'>{label}</div>"
            f"<div class='app-card-sub'>{path}</div></div>",
            unsafe_allow_html=True,
        )
