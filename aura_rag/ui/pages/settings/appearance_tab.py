"""Appearance settings tab."""

import streamlit as st

from ....rag_application import RAGApplication
from ...i18n import t


def render_appearance_tab(app: RAGApplication):
    """Appearance settings tab."""
    st.markdown(f"##### 🎨 {t('tab_appearance')}")
    theme_options = {"dark": t("theme_dark"), "light": t("theme_light")}
    chosen_theme = st.radio(
        t("theme_label"),
        options=list(theme_options.keys()),
        format_func=lambda k: theme_options[k],
        index=list(theme_options.keys()).index(app.config.config.get("theme", "dark")),
        key="theme_settings_radio",
    )
    if chosen_theme != st.session_state.get("theme", "dark"):
        st.session_state.theme = chosen_theme
        app.config.config["theme"] = chosen_theme
        app.config.save_config()
        st.rerun()
