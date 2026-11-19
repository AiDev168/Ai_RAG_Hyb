# AURA RAG — modular layout

Run: `pip install -r requirements.txt` then `streamlit run app.py`

Behaviour is identical to the single-file `aura_rag7.py`; the code was moved, not rewritten
(one exception: `extractive_answer()` now takes `language` as an argument instead of reading
Streamlit state, so the backend no longer depends on the UI).

```
app.py                       entry point
aura_rag/
  __init__.py                installs the network guard FIRST (keep this import order)
  network.py                 Offline/Online mode + socket guard
  dependencies.py            optional-package imports, IMPORT_ERRORS, SUPPORTED_EXTENSIONS
  constants.py               presets, provider list, icons, "don't know" messages
  config.py                  Config + defaults (config.json)
  ocr.py                     OCR text cleaning / language resolution
  text_utils.py              FTS5 helpers, entities, citations, evidence
  persistent_store.py        SQLite: chunks, FTS5, manifest, conversations, answer cache
  document_processor.py      loading (incl. OCR) + splitting
  vector_store.py            Chroma
  retriever.py               hybrid search, rerank, MMR, graph
  llm_manager.py             GGUF + OpenAI-compatible backends
  document_manager.py        documents folder
  rag_application.py         orchestrates everything above
  folder_indexer.py          background bulk indexing
  ui/
    translations.py  i18n.py  theme.py  components.py
    chat_render.py   chat_export.py     main_view.py   (sidebar + navigation)
    pages/
      chat.py  documents.py  status.py
      settings/  __init__.py + llm / retrieval / embeddings / chat /
                 appearance / storage / language tabs
```

Dependency direction: `ui` → core modules → `dependencies`/`constants`. Core never imports `ui`.
