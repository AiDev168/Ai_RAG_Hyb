"""Embeddings & chunking settings tab."""

import streamlit as st
from pathlib import Path

from ....config import resolve_local_model
from ....dependencies import OCR_AVAILABLE, Pytesseract
from ....network import is_offline_mode
from ....ocr import OCR_LANG_INSTALL_HINT
from ....rag_application import RAGApplication
from ...i18n import current_lang


def render_embeddings_tab(app: RAGApplication):
    """Embeddings & chunking settings tab."""
    embedding_options = [
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/all-mpnet-base-v2",
        "sentence-transformers/paraphrase-MiniLM-L6-v2",
        "BAAI/bge-small-en-v1.5",
        "BAAI/bge-base-en-v1.5",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        "intfloat/multilingual-e5-base",
        "intfloat/multilingual-e5-large",
        "heydariAI/persian-embeddings",
    ]
    multilingual_models = {
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        "intfloat/multilingual-e5-base",
        "intfloat/multilingual-e5-large",
        "heydariAI/persian-embeddings",
    }
    current_embedding = app.config.config["embedding_model"]
    default_index = (
        embedding_options.index(current_embedding)
        if current_embedding in embedding_options
        else 0
    )

    if current_lang() == "fa" and current_embedding not in multilingual_models:
        st.info(
            "برای بهترین نتیجه با اسناد فارسی، یکی از مدل‌های چندزبانه (مثل "
            "paraphrase-multilingual-mpnet-base-v2 یا multilingual-e5-base) را انتخاب کنید."
        )

    embedding_model = st.selectbox(
        "Embedding Model", embedding_options, index=default_index,
        help="Models marked multilingual below support Persian and other "
             "non-English scripts; the plain English ones do not.",
    )
    local_embedding_path = st.text_input(
        "Local embedding model folder (optional — overrides the list above)",
        value=current_embedding if Path(current_embedding).expanduser().is_dir() else "",
        placeholder=str(app.config.models_dir / "embeddings" / "my-model"),
        help="Offline mode never downloads models; online mode downloads names from the list "
             "automatically. Copy a sentence-transformers model folder "
             f"into {app.config.models_dir / 'embeddings'} (named after the model, e.g. "
             "'paraphrase-multilingual-mpnet-base-v2') or point to any folder here.",
    )
    if local_embedding_path.strip():
        embedding_model = local_embedding_path.strip()
    _resolved_embedding = resolve_local_model(embedding_model, app.config.models_dir, "embeddings")
    if Path(_resolved_embedding).is_dir():
        st.caption(f"✅ Found locally: `{_resolved_embedding}`")
    elif is_offline_mode():
        st.caption(
            "⚠️ Not found locally. In offline mode it only loads if it is already in the HuggingFace "
            f"cache; otherwise copy the model folder into `{app.config.models_dir / 'embeddings'}` "
            "or switch to Online mode."
        )
    else:
        st.caption("🌐 Not found locally — it will be downloaded from HuggingFace the next time it is loaded (online mode).")
    if embedding_model in multilingual_models:
        st.caption("🌐 Multilingual model — supports Persian and other non-English scripts.")
    if embedding_model != current_embedding:
        st.caption(
            "⚠️ Changing the embedding model does not re-embed existing chunks. "
            "Clear the knowledge base and re-add documents after switching."
        )

    col1, col2 = st.columns(2)

    with col1:
        chunk_size = st.number_input(
            "Chunk Size",
            min_value=100,
            max_value=4000,
            value=app.config.config["chunk_size"],
            step=100,
        )

    with col2:
        chunk_overlap = st.number_input(
            "Chunk Overlap",
            min_value=0,
            max_value=500,
            value=app.config.config["chunk_overlap"],
            step=50,
        )

    st.markdown("##### 🧱 Chunking strategy")
    chunking_options = {"fixed": "Fixed size (character count)", "structural": "Structure-aware (headings & paragraphs first)"}
    chunking_strategy = st.selectbox(
        "How documents are split into chunks",
        options=list(chunking_options.keys()),
        format_func=lambda k: chunking_options[k],
        index=list(chunking_options.keys()).index(app.config.config.get("chunking_strategy", "fixed")),
        help="Structure-aware tries to split on headings and paragraph breaks before "
             "falling back to plain character counts, so a chunk is more likely to be "
             "a coherent section instead of an arbitrary slice through a sentence. "
             "Best for well-formatted documents (Markdown, docx, structured PDFs).",
    )

    st.markdown("##### 🔎 OCR for scanned PDFs and images (Persian supported)")
    if not OCR_AVAILABLE:
        st.info(
            "OCR isn't available — install `pymupdf` and `pytesseract` "
            "(plus the `tesseract-ocr` system package) to enable it. "
            "Without it, scanned/image-only PDFs will fail to index with "
            "'no usable text' instead of being read via OCR."
        )
    else:
        if st.button("🩺 Run OCR diagnostic check"):
            # Importing pytesseract only confirms the *Python package* is
            # present — it says nothing about whether the tesseract-ocr
            # *system binary* it wraps is actually installed and on
            # PATH, which is the far more common way OCR silently fails.
            # get_tesseract_version() is the one call that actually
            # invokes the binary, so it's the real test.
            try:
                version = Pytesseract.get_tesseract_version()
                st.success(f"✅ tesseract-ocr engine found (version {version}).")
                langs_configured = app.config.config.get("ocr_languages", "fas+eng")
                try:
                    available_langs = set(Pytesseract.get_languages())
                    for lang in langs_configured.split("+"):
                        lang = lang.strip()
                        if lang and lang not in available_langs:
                            st.warning(
                                f"⚠️ Language pack '{lang}' is not installed for tesseract "
                                f"(available: {', '.join(sorted(available_langs)) or 'none detected'}). "
                                f"Install it, e.g. 'apt install tesseract-ocr-{lang}'. "
                                + (OCR_LANG_INSTALL_HINT if lang == "fas" else "")
                            )
                        elif lang:
                            st.caption(f"✅ Language pack '{lang}' is installed.")
                except Exception:
                    pass
            except Exception as e:
                st.error(
                    f"❌ The tesseract-ocr Python package (pytesseract) imported fine, but "
                    f"calling the actual OCR engine failed: {type(e).__name__}: {e}\n\n"
                    f"This almost always means the tesseract-ocr **system package** isn't "
                    f"installed or isn't on PATH — install it separately from the Python "
                    f"packages (e.g. 'apt install tesseract-ocr' on Debian/Ubuntu, "
                    f"'brew install tesseract' on macOS), then restart the app."
                )
    ocr_enabled = st.toggle(
        "Fall back to OCR when a PDF has no extractable text layer",
        value=app.config.config.get("ocr_enabled", True) and OCR_AVAILABLE,
        disabled=not OCR_AVAILABLE,
        help="Only used when normal text extraction finds nothing — most PDFs have a "
             "real text layer and never touch OCR, so this costs nothing for them.",
    )
    _ocr_presets = {
        "fas+eng": "Persian + English (recommended for Persian documents)",
        "fas": "Persian only",
        "eng": "English only",
    }
    _ocr_current = app.config.config.get("ocr_languages", "fas+eng")
    _ocr_keys = list(_ocr_presets.keys()) + ["custom"]
    _ocr_choice = st.selectbox(
        "OCR language",
        options=_ocr_keys,
        index=_ocr_keys.index(_ocr_current) if _ocr_current in _ocr_presets else len(_ocr_keys) - 1,
        format_func=lambda k: _ocr_presets.get(k, "Custom (tesseract codes)"),
        disabled=not (ocr_enabled and OCR_AVAILABLE),
        help="Needs the matching tesseract language pack ('fas' = Persian). "
             "Packs that aren't installed are skipped with a warning in the log.",
    )
    if _ocr_choice == "custom":
        ocr_languages = st.text_input(
            "Custom OCR language(s) (tesseract codes, + separated)",
            value=_ocr_current,
            disabled=not (ocr_enabled and OCR_AVAILABLE),
            help="e.g. 'fas+ara+eng'.",
        )
    else:
        ocr_languages = _ocr_choice
    ocr_dpi = st.number_input(
        "OCR render resolution (DPI)",
        min_value=100, max_value=400, step=25,
        value=int(app.config.config.get("ocr_dpi", 250)),
        disabled=not (ocr_enabled and OCR_AVAILABLE),
        help="Higher is more accurate for Persian (small dots and joins) but slower "
             "and uses more memory. 250–300 is a good range for scanned PDFs. "
             "Image files (PNG/JPG/TIFF...) are OCR'd at their own resolution.",
    )

    if st.button("💾 Save Embedding Settings", type="primary", use_container_width=True):
        app.config.config["embedding_model"] = embedding_model
        app.config.config["chunk_size"] = chunk_size
        app.config.config["chunk_overlap"] = chunk_overlap
        app.config.config["chunking_strategy"] = chunking_strategy
        app.config.config["ocr_enabled"] = ocr_enabled and OCR_AVAILABLE
        app.config.config["ocr_languages"] = ocr_languages.strip() or "fas+eng"
        app.config.config["ocr_dpi"] = int(ocr_dpi)
        app.config.save_config()
        # Rebuild the splitter immediately so new chunk settings actually apply.
        app.doc_processor._build_splitter()
        st.success("Settings saved and applied to new documents.")
