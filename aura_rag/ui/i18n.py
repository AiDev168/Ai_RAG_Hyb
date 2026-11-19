"""Translation lookup and text-direction helpers."""

import streamlit as st

from .translations import RTL_LANGS, TRANSLATIONS


def t(key: str) -> str:
    """Translate a UI string key into the currently active language,
    falling back to English if the key or language is missing."""
    lang = st.session_state.get("language", "en")
    return TRANSLATIONS.get(lang, TRANSLATIONS["en"]).get(key, TRANSLATIONS["en"].get(key, key))


def current_lang() -> str:
    return st.session_state.get("language", "en")


def is_rtl_lang(lang: str) -> bool:
    return lang in RTL_LANGS


_PERSIAN_ARABIC_RANGES = (
    (0x0600, 0x06FF),  # Arabic
    (0x0750, 0x077F),  # Arabic Supplement
    (0x08A0, 0x08FF),  # Arabic Extended-A
    (0xFB50, 0xFDFF),  # Arabic Presentation Forms-A
    (0xFE70, 0xFEFF),  # Arabic Presentation Forms-B
)


def detect_text_direction(text: str) -> str:
    """Heuristically detect whether a piece of text (e.g. a chat message or
    a document chunk) is predominantly RTL (Persian/Arabic script) or LTR,
    so mixed-language chat history can render each bubble with the correct
    alignment instead of forcing one direction on everything."""
    if not text:
        return "ltr"
    rtl_count = 0
    ltr_count = 0
    for ch in text:
        code = ord(ch)
        if any(lo <= code <= hi for lo, hi in _PERSIAN_ARABIC_RANGES):
            rtl_count += 1
        elif ch.isalpha():
            ltr_count += 1
    if rtl_count == 0 and ltr_count == 0:
        return "ltr"
    return "rtl" if rtl_count >= ltr_count else "ltr"
