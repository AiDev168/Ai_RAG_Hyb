# AURA RAG setup and launch

This guide covers a clean install on Windows or Linux. Python packages live in a per-project `.venv`; documents, settings, the Chroma database, and downloaded models live in the user's home folder.

## Requirements

- Windows 10/11 or a current Linux distribution
- Python 3.11 (the checked-in lock file targets Python 3.11)
- Git
- `uv` for environment creation and dependency installation
- Internet access for the first dependency install and first model download

Install `uv` using the instructions at <https://docs.astral.sh/uv/getting-started/installation/>. Install Git and Python 3.11 if they are not already available. `uv` can also install the project Python with `uv python install 3.11`.

## Clone, install, and launch

Run these commands from a terminal. Replace the repository URL if using a fork.

### Windows PowerShell

```powershell
git clone https://github.com/AiDev168/Ai_RAG_Hyb.git
cd Ai_RAG_Hyb
uv python install 3.11
uv sync --locked
uv run streamlit run app.py
```

### Linux

```sh
git clone https://github.com/AiDev168/Ai_RAG_Hyb.git
cd Ai_RAG_Hyb
uv python install 3.11
uv sync --locked
uv run streamlit run app.py
```

Open the local address printed by Streamlit, normally <http://localhost:8501>. Stop the server with `Ctrl+C`. To start it again, run `uv run streamlit run app.py` from the repository directory.

`uv sync --locked` creates `.venv` and installs exact versions from `uv.lock`. Run it again after pulling a change to the lock file. Do not copy `.venv` between computers or operating systems; clone the repository and create a fresh environment on each system.

## Install the embedding model

The app starts in **Offline** mode by default. This prevents unexpected network access, but the embedding model is a separate download from the Python packages. If it is missing, the app now opens and shows a warning so you can fix the model from Settings.

For Persian and multilingual documents, download this supported model once while the computer has internet access:

The model weights are about 450 MB. The commands below fetch only the Safetensors weights and tokenizer/config files needed by the app, not the repository's alternative ONNX/OpenVINO copies.

### Windows PowerShell

```powershell
uv run hf download sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 config.json config_sentence_transformers.json model.safetensors modules.json sentence_bert_config.json sentencepiece.bpe.model special_tokens_map.json tokenizer.json tokenizer_config.json 1_Pooling/config.json --local-dir "$env:USERPROFILE\.local_rag_app\models\embeddings\paraphrase-multilingual-MiniLM-L12-v2" --max-workers 4
```

### Linux

```sh
uv run hf download sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 config.json config_sentence_transformers.json model.safetensors modules.json sentence_bert_config.json sentencepiece.bpe.model special_tokens_map.json tokenizer.json tokenizer_config.json 1_Pooling/config.json --local-dir "$HOME/.local_rag_app/models/embeddings/paraphrase-multilingual-MiniLM-L12-v2" --max-workers 4
```

Then open **Settings → Embeddings & Chunking**, choose `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, and click **Save Embedding Settings**. The app loads it from the local folder; it can remain in Offline mode. If you prefer to download the currently selected model from inside the app, switch the sidebar to **Online**, wait for the model to load, and switch back to **Offline** if desired.

To use a model folder copied from another computer, copy the complete folder (including its config, tokenizer, and model weight files) under:

```text
<user-home>/.local_rag_app/models/embeddings/<model-folder>
```

Then select its corresponding model name in Embeddings & Chunking, or enter the full local folder path in the local model field. A Hugging Face cache entry alone is not guaranteed to be portable; the `hf download --local-dir` commands above create a self-contained folder that is straightforward to copy.

Changing the embedding model changes how text is represented. If documents were already indexed, use the app's **Clear knowledge base** action and re-add those documents after changing models. Do not search old vectors with a newly selected model.

## Per-user data and optional features

The app stores private runtime data outside the clone:

| Data | Windows | Linux |
| --- | --- | --- |
| Documents, config, Chroma database, models | `%USERPROFILE%\.local_rag_app\` | `~/.local_rag_app/` |

The config may contain API credentials. Do not commit or share it. Back up this data directory if you need to preserve a local library; it is not part of the Git repository.

Optional additional file loaders can be installed with `uv pip install -r requirements-office.txt`. The local GGUF/llama.cpp backend can be installed with `uv pip install -r requirements-local-llm.txt`; available builds depend on OS, Python version, CPU, and compiler. Other configured LLM backends do not require llama.cpp.

OCR for scanned PDFs and images additionally needs the Tesseract system application and language packs. The core app and ordinary text-based PDF ingestion do not require Tesseract.

## Troubleshooting

- **Model not found / Hugging Face says outgoing traffic is disabled:** keep Offline mode and run the matching `uv run hf download ... --local-dir ...` command above, or switch to Online mode so the app downloads the selected model. Save the embedding settings after copying a model folder.
- **Python/dependency mismatch:** from the repository root, run `uv sync --locked`, then launch with `uv run streamlit run app.py`.
- **Port 8501 is occupied:** run `uv run streamlit run app.py --server.port 8502` and open <http://localhost:8502>.
- **Different OS:** use a fresh clone and `uv sync --locked`; never move `.venv` from another OS.
