"""Language settings tab."""

import streamlit as st

from ....rag_application import RAGApplication
from ...i18n import current_lang, t


def render_language_tab(app: RAGApplication):
    """Language settings tab."""
    st.markdown(f"##### {t('language_label')}")
    st.caption(
        "Switching language also switches the app's text direction (RTL for "
        "Persian) and the language the assistant is asked to answer in."
        if current_lang() != "fa" else
        "تغییر زبان، جهت متن برنامه را نیز تغییر می‌دهد (راست‌به‌چپ برای فارسی) "
        "و زبانی که از دستیار خواسته می‌شود پاسخ دهد."
    )
    lang_options = {"en": "🇬🇧 English", "fa": "🇮🇷 فارسی (Persian)"}
    chosen = st.radio(
        t("language_label"),
        options=list(lang_options.keys()),
        format_func=lambda k: lang_options[k],
        index=list(lang_options.keys()).index(current_lang()),
        key="language_settings_radio",
    )
    if chosen != st.session_state.language:
        st.session_state.language = chosen
        app.config.config["language"] = chosen
        app.config.save_config()
        st.rerun()

    st.markdown("---")
    st.caption(
        "Tip: for Persian documents, also pick a multilingual embedding model "
        "in the Embeddings & Chunking tab and keep hybrid (BM25) search on in "
        "the Retrieval tab — it helps a lot with exact Persian terms and names."
        if current_lang() != "fa" else
        "نکته: برای اسناد فارسی، در تب امبدینگ و تکه‌بندی یک مدل چندزبانه انتخاب کنید "
        "و در تب بازیابی، جستجوی ترکیبی (BM25) را روشن نگه دارید — این کار به تشخیص "
        "دقیق اصطلاحات و نام‌های فارسی کمک زیادی می‌کند."
    )
