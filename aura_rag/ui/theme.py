"""Theme palettes and CSS injection."""

import streamlit as st

from .i18n import current_lang, is_rtl_lang


# --------------------------------------------------------------------------
# UI theming
# Direction: a quiet "reading room" look. Cool slate neutrals, ONE accent
# (deep teal), hairline borders instead of shadows, no gradients, no purple
# sidebar, no banner. The accent is reserved for the primary action, the
# active nav item, and links/focus, so it actually means something.
# --------------------------------------------------------------------------
_PALETTES = {
    "light": {
        "bg": "#F4F6F8", "surface": "#FFFFFF", "sunken": "#EBEFF3",
        "border": "#D9DFE6", "border_strong": "#BAC4CF",
        "text": "#16202A", "muted": "#5B6875",
        "accent": "#0E6F66", "accent_soft": "#DDF1EE", "on_accent": "#FFFFFF",
        "code_bg": "#EBEFF3", "danger": "#B42318",
    },
    "dark": {
        "bg": "#0E1317", "surface": "#151C22", "sunken": "#1B242C",
        "border": "#27323C", "border_strong": "#3A4854",
        "text": "#E6ECF1", "muted": "#93A1AE",
        "accent": "#5CC8B8", "accent_soft": "#17332F", "on_accent": "#08201C",
        "code_bg": "#0B1014", "danger": "#F97066",
    },
}

_CSS_TEMPLATE = """
<style>
:root {
  --bg: __bg__; --surface: __surface__; --sunken: __sunken__;
  --border: __border__; --border-strong: __border_strong__;
  --text: __text__; --muted: __muted__;
  --accent: __accent__; --accent-soft: __accent_soft__; --on-accent: __on_accent__;
  --danger: __danger__;
  --r-sm: 6px; --r-md: 10px; --r-lg: 14px;
}

/* ---------- base ---------- */
html, body, .stApp { font-family: __font__; }
.stApp {
  background: var(--bg);
  color: var(--text);
  direction: __dir__;
}
.stApp :where(p, li, label, h1, h2, h3, h4, h5, h6) { font-family: __font__; }
/* Streamlit draws its icons (expander arrows, etc.) with an icon FONT inside a
   <span>. Never override that span's font, or the icon name shows as plain text
   ("arrow_right"). Re-assert the icon font in case anything else touches it. */
.stApp [data-testid="stIconMaterial"],
.stApp span[class*="material" i],
.stApp .material-icons,
.stApp .material-symbols-rounded {
  font-family: "Material Symbols Rounded", "Material Symbols Outlined", "Material Icons" !important;
  font-feature-settings: "liga"; -webkit-font-feature-settings: "liga";
  letter-spacing: normal; text-transform: none; direction: ltr;
}
/* Only set color on text elements we own; never on generic div/span, which
   would override Streamlit's own colors (badges, links, disabled states). */
.stMarkdown, .stMarkdown p, .stMarkdown li, label, h1, h2, h3, h4, h5 { color: var(--text); }
.stCaption, [data-testid="stCaptionContainer"] { color: var(--muted) !important; }
.stMarkdown, .stMarkdown p, .stMarkdown li, h1, h2, h3, h4, h5, h6,
label, .stAlert { text-align: __align__; }
.block-container { padding-top: 2rem; padding-bottom: 5rem; max-width: 1100px; }
h1, h2, h3 { font-weight: 650 !important; letter-spacing: -0.015em; }
h5 { font-weight: 600 !important; margin-top: 0.5rem; }
a { color: var(--accent) !important; }
hr { border-color: var(--border) !important; }
header[data-testid="stHeader"] { background: transparent; }

/* visible keyboard focus everywhere */
:where(button, input, textarea, select, [role="tab"], [role="radio"]):focus-visible {
  outline: 2px solid var(--accent) !important; outline-offset: 2px;
}

/* ---------- page header ---------- */
.page-head { padding: 0 0 14px 0; margin-bottom: 18px; border-bottom: 1px solid var(--border); }
.page-head h1 { font-size: 1.7rem; margin: 0 0 4px 0; padding: 0; color: var(--text); text-align: __align__; }
.page-head p { margin: 0; font-size: 0.95rem; color: var(--muted); max-width: 62ch; text-align: __align__; }

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"] {
  background: var(--surface);
  border-__side__: 1px solid var(--border);
  direction: __dir__;
}
section[data-testid="stSidebar"] .block-container { padding-top: 1.2rem; }
.brand { display: flex; align-items: center; gap: 10px; margin-bottom: 2px; }
.brand-mark {
  width: 32px; height: 32px; border-radius: 8px; background: var(--accent);
  color: var(--on-accent); display: grid; place-items: center; font-size: 1rem;
}
.brand-name { font-size: 1.05rem; font-weight: 650; color: var(--text); }

/* nav: the navigation st.radio has key="nav_radio" */
.st-key-nav_radio div[role="radiogroup"] { gap: 2px; }
.st-key-nav_radio label {
  padding: 8px 12px; border-radius: var(--r-md); width: 100%; cursor: pointer;
  border: 1px solid transparent;
}
.st-key-nav_radio label:hover { background: var(--sunken); }
.st-key-nav_radio label > div:first-child { display: none; }          /* hide radio dot */
.st-key-nav_radio label:has(input:checked) {
  background: var(--accent-soft); border-color: transparent;
}
.st-key-nav_radio label:has(input:checked) p { color: var(--accent); font-weight: 650; }
.st-key-nav_radio label p { font-weight: 500; margin: 0; }

/* network switch as a segmented control */
.st-key-network_mode_radio div[role="radiogroup"] {
  background: var(--sunken); border-radius: var(--r-md); padding: 3px; gap: 3px; width: 100%;
}
.st-key-network_mode_radio label {
  flex: 1; justify-content: center; padding: 5px 8px; border-radius: 8px; cursor: pointer; margin: 0;
}
.st-key-network_mode_radio label > div:first-child { display: none; }
.st-key-network_mode_radio label:has(input:checked) {
  background: var(--surface); box-shadow: 0 0 0 1px var(--border-strong);
}
.st-key-network_mode_radio label p { margin: 0; font-size: 0.85rem; font-weight: 600; }

/* sidebar metrics: flat rows, not boxes */
section[data-testid="stSidebar"] [data-testid="stMetric"] {
  background: transparent; border: none; border-bottom: 1px solid var(--border);
  border-radius: 0; padding: 8px 0;
}
section[data-testid="stSidebar"] [data-testid="stMetricValue"] { font-size: 1.05rem; }

/* ---------- buttons ---------- */
.stButton > button, .stDownloadButton > button {
  border-radius: var(--r-md); font-weight: 600;
  border: 1px solid var(--border-strong); background: var(--surface); color: var(--text);
  transition: background-color .12s ease, border-color .12s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
  border-color: var(--accent); color: var(--accent); background: var(--surface);
}
.stButton > button[kind="primary"] {
  background: var(--accent); border-color: var(--accent); color: var(--on-accent);
}
.stButton > button[kind="primary"]:hover { filter: brightness(1.08); color: var(--on-accent); }
.stButton > button:disabled { opacity: .45; }

/* ---------- inputs ---------- */
.stTextInput input, .stNumberInput input, .stTextArea textarea,
div[data-baseweb="select"] > div, div[data-baseweb="input"] {
  background-color: var(--surface) !important; color: var(--text) !important;
  border-color: var(--border-strong) !important; border-radius: var(--r-md) !important;
  direction: __dir__; text-align: __align__;
}
.stTextInput input:focus, .stTextArea textarea:focus { border-color: var(--accent) !important; }

/* ---------- tabs ---------- */
button[data-baseweb="tab"] { color: var(--muted); font-weight: 500; padding-left: 14px; padding-right: 14px; }
button[data-baseweb="tab"][aria-selected="true"] { color: var(--text); font-weight: 650; }
div[data-baseweb="tab-highlight"] { background-color: var(--accent) !important; height: 2px !important; }
div[data-baseweb="tab-border"] { background-color: var(--border) !important; }

/* ---------- expanders ---------- */
details {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: var(--r-md); direction: __dir__;
}
details summary { font-weight: 500; }

/* ---------- uploader ---------- */
[data-testid="stFileUploaderDropzone"] {
  background: var(--surface); border: 1.5px dashed var(--border-strong) !important;
  border-radius: var(--r-lg);
}
[data-testid="stFileUploaderDropzone"]:hover { border-color: var(--accent) !important; }

/* ---------- metrics (main area) ---------- */
[data-testid="stMetric"] {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: var(--r-md); padding: 12px 16px; text-align: __align__;
}
[data-testid="stMetricLabel"] { color: var(--muted) !important; }
[data-testid="stMetricValue"] { color: var(--text) !important; font-weight: 650; }

/* ---------- alerts ---------- */
[data-testid="stAlert"] { border-radius: var(--r-md); border: 1px solid var(--border); }

/* ---------- code ---------- */
.stCodeBlock, pre, code {
  background-color: __code_bg__ !important; color: var(--text) !important;
  direction: ltr; text-align: left; border-radius: var(--r-sm);
}

/* ---------- chat ---------- */
[data-testid="stChatMessage"] {
  background: transparent; border-radius: var(--r-lg);
  padding: 12px 14px; margin-bottom: 4px;
}
/* user turns sit on a sunken surface; assistant turns are plain page */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
  background: var(--sunken);
}
[data-testid="stChatMessage"] .stMarkdown p { line-height: 1.7; max-width: 75ch; }
[data-testid="stChatInput"] {
  border: 1px solid var(--border-strong); border-radius: var(--r-lg); background: var(--surface);
}
[data-testid="stChatInput"]:focus-within { border-color: var(--accent); }
[data-testid="stChatInput"] textarea { background: transparent !important; }

/* ---------- reusable pieces ---------- */
.app-card {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: var(--r-md); padding: 12px 16px; margin-bottom: 8px; text-align: __align__;
}
.app-card-title { font-weight: 600; font-size: .95rem; color: var(--text); }
.app-card-sub { font-size: .8rem; color: var(--muted); }
.app-pill {
  display: inline-block; background: var(--sunken); color: var(--muted);
  border: 1px solid var(--border); border-radius: 999px;
  padding: 2px 10px; font-size: .75rem; font-weight: 600;
}
.app-pill.accent { background: var(--accent-soft); color: var(--accent); border-color: transparent; }
.doc-row {
  display: flex; align-items: center; gap: 12px; background: var(--surface);
  border: 1px solid var(--border); border-radius: var(--r-md); padding: 10px 14px;
}
.doc-row .doc-name { font-weight: 600; flex: 1; overflow-wrap: anywhere; }
.doc-row .doc-size { color: var(--muted); font-size: .85rem; white-space: nowrap; }

.empty-state {
  text-align: center; padding: 40px 24px; background: var(--surface);
  border: 1.5px dashed var(--border-strong); border-radius: var(--r-lg); color: var(--muted);
}
.empty-state .emoji {
  width: 52px; height: 52px; margin: 0 auto 12px; border-radius: 50%;
  background: var(--sunken); display: grid; place-items: center; font-size: 1.5rem;
}
.empty-state .es-title { font-weight: 650; color: var(--text); margin-bottom: 4px; font-size: 1.02rem; }
.empty-state .es-sub { max-width: 46ch; margin: 0 auto; font-size: .92rem; }

/* primary buttons + active chat use the theme accent (Streamlit's default primary is red) */
button[data-testid="stBaseButton-primary"], button[data-testid="stBaseButton-primaryFormSubmit"] {
  background: var(--accent) !important; border-color: var(--accent) !important; color: var(--on-accent) !important; }
button[data-testid="stBaseButton-primary"] p, button[data-testid="stBaseButton-primaryFormSubmit"] p {
  color: var(--on-accent) !important; }
button[data-testid="stBaseButton-primary"]:hover { filter: brightness(1.08); }
[class*="st-key-convactive_"] button {
  background: var(--accent-soft) !important; border-color: transparent !important; }
[class*="st-key-convactive_"] button p { color: var(--accent) !important; font-weight: 650; }
/* ---------- AURA pieces: chips, source cards, audit rows ---------- */
.chip { display: inline-block; padding: 2px 10px; margin: 0 6px 4px 0; border-radius: 999px; font-size: .74rem;
  font-weight: 600; border: 1px solid var(--border-strong); background: var(--sunken); color: var(--muted); }
.chip.ok { background: rgba(34,197,94,.14); border-color: rgba(34,197,94,.45); color: #16a34a; }
.chip.info { background: var(--accent-soft); border-color: transparent; color: var(--accent); }
.chip.warn { background: rgba(245,158,11,.14); border-color: rgba(245,158,11,.5); color: #d97706; }
.chip.bad { background: rgba(239,68,68,.14); border-color: rgba(239,68,68,.45); color: var(--danger); }
.src-card { border: 1px solid var(--border); border-radius: var(--r-md); padding: .65rem .9rem; margin-bottom: .6rem;
  background: var(--surface); }
.src-card.cited { border-inline-start: 4px solid var(--accent); }
.src-head { display: flex; align-items: center; gap: .55rem; flex-wrap: wrap; }
.src-num { display: inline-flex; align-items: center; justify-content: center; min-width: 1.5rem; height: 1.5rem;
  border-radius: 8px; background: var(--accent); color: var(--on-accent); font-weight: 700; font-size: .8rem; }
.src-title { font-weight: 600; font-size: .92rem; overflow-wrap: anywhere; }
.src-meta { color: var(--muted); font-size: .76rem; margin: .25rem 0 .35rem 0; }
.bar { height: 5px; border-radius: 999px; background: var(--sunken); overflow: hidden; }
.bar > span { display: block; height: 100%; background: var(--accent); }
.src-card details { margin-top: .5rem; background: transparent; border: none; }
.src-card summary { cursor: pointer; font-size: .8rem; color: var(--muted); }
.src-text { margin-top: .4rem; max-height: 220px; overflow: auto; font-size: .84rem; line-height: 1.5;
  padding: .5rem .7rem; border-radius: 8px; background: var(--sunken); }
.aud { border-inline-start: 4px solid var(--border-strong); padding: .35rem .7rem; margin: .35rem 0;
  border-radius: 6px; background: var(--sunken); font-size: .88rem; }
.aud.ok { border-color: #16a34a; } .aud.info { border-color: var(--accent); }
.aud.warn { border-color: #d97706; } .aud.bad { border-color: var(--danger); }
.aud small { color: var(--muted); }
section[data-testid="stSidebar"] .stButton > button { justify-content: flex-start; text-align: start; }

@media (max-width: 640px) {
  .block-container { padding-left: 1rem; padding-right: 1rem; }
  .page-head h1 { font-size: 1.4rem; }
}
</style>
"""


def inject_custom_css():
    rtl = is_rtl_lang(current_lang())
    theme = st.session_state.get("theme", "dark")
    pal = _PALETTES.get(theme, _PALETTES["dark"])
    # System fonts only — nothing is fetched from the web. Vazirmatn / Inter
    # are used if installed locally.
    font = (
        "'Vazirmatn', 'Segoe UI', Tahoma, 'Noto Sans Arabic', system-ui, sans-serif"
        if rtl else
        "'Inter', system-ui, -apple-system, 'Segoe UI', Roboto, Tahoma, 'Noto Sans Arabic', sans-serif"
    )
    css = _CSS_TEMPLATE
    values = dict(pal, font=font, dir="rtl" if rtl else "ltr",
                  align="right" if rtl else "left",
                  side="left" if rtl else "right")
    for key, val in values.items():
        css = css.replace(f"__{key}__", val)
    st.markdown(css, unsafe_allow_html=True)
