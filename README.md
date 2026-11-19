# AURA RAG

## Run locally

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and Python 3.11, then from the repository root run:

```sh
uv sync --locked
uv run streamlit run app.py
```

The same commands work in PowerShell, Windows Command Prompt, and Linux shells. `uv sync` creates the project environment in `.venv` and installs the exact versions recorded in `uv.lock`.

Optional features:

```sh
uv pip install -r requirements-office.txt  # additional Office/markup document formats
uv pip install -r requirements-local-llm.txt  # requires the project's environment to be active
```

The application can use its other configured LLM backends without `llama-cpp-python`. Scanned document OCR also requires the Tesseract system application and language data; these are separate from Python packages.

For clean Windows/Linux setup, embedding model download, data locations, and troubleshooting, see [docs/SETUP.md](docs/SETUP.md).

## CI

GitHub Actions installs the locked dependencies on both the latest Ubuntu and Windows runners for every push and pull request. To refresh the dependency lock after editing `pyproject.toml`, run `uv lock` and commit both `pyproject.toml` and `uv.lock`.

## Project layout

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
