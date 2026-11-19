"""Static constants: LLM backend modes, provider presets, retrieval presets, icons, messages."""

from pathlib import Path


LLM_BACKEND_MODES = {
    "gguf_only": "Offline GGUF only (default, fully private)",
    "gguf_then_openai": "Offline GGUF first, OpenAI-compatible API as fallback",
    "openai_only": "OpenAI-compatible API only",
}

# One-click presets for the OpenAI-compatible backend. Anything that speaks
# the OpenAI /v1/chat/completions protocol works — hosted services and
# self-hosted servers alike. "key_required" only affects the hint shown in
# Settings: when no key is set and a base URL is given, a placeholder key is
# sent automatically, because local servers generally ignore it.
OPENAI_PROVIDER_PRESETS = {
    "openai": {"label": "OpenAI (api.openai.com)", "base_url": "", "model": "gpt-4o-mini", "key_required": True},
    "openrouter": {"label": "OpenRouter", "base_url": "https://openrouter.ai/api/v1", "model": "openai/gpt-4o-mini", "key_required": True},
    "groq": {"label": "Groq", "base_url": "https://api.groq.com/openai/v1", "model": "llama-3.3-70b-versatile", "key_required": True},
    "together": {"label": "Together AI", "base_url": "https://api.together.xyz/v1", "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo", "key_required": True},
    "ollama": {"label": "Ollama (local)", "base_url": "http://localhost:11434/v1", "model": "llama3.2", "key_required": False},
    "lmstudio": {"label": "LM Studio (local)", "base_url": "http://localhost:1234/v1", "model": "local-model", "key_required": False},
    "vllm": {"label": "vLLM (self-hosted)", "base_url": "http://localhost:8000/v1", "model": "", "key_required": False},
    "llamacpp": {"label": "llama.cpp server (local)", "base_url": "http://localhost:8080/v1", "model": "local-model", "key_required": False},
    "custom": {"label": "Custom OpenAI-compatible endpoint", "base_url": "", "model": "", "key_required": False},
}


EXT_ICONS = {
    ".pdf": "📕", ".txt": "📃", ".docx": "📘", ".doc": "📘",
    ".csv": "📊", ".xlsx": "📗", ".xls": "📗",
    ".pptx": "📙", ".ppt": "📙", ".md": "📝",
    ".html": "🌐", ".htm": "🌐", ".epub": "📖", ".odt": "📘",
    ".rtf": "📄", ".eml": "✉️", ".msg": "✉️", ".json": "🔧",
}

# Bump this whenever the answering prompt / "don't know" behaviour changes.
# It is part of the answer-cache fingerprint, so older cached answers (for
# example ones produced from the model's general knowledge) are invalidated
# automatically instead of being served again.
PROMPT_VERSION = "strict-docs-only-v3-cited"

# Retrieval presets (applied from Settings -> Retrieval).
PRESETS = {
    "Fast": {"final_k": 4, "neighbors": 0, "max_context_chars": 4000, "rerank_candidates": 15},
    "Balanced": {"final_k": 5, "neighbors": 1, "max_context_chars": 6000, "rerank_candidates": 20},
    "Thorough": {"final_k": 8, "neighbors": 1, "max_context_chars": 12000, "rerank_candidates": 40},
}

# Fixed replies used when the answer is not in the documents.
NO_ANSWER_MESSAGES = {
    "en": "I don't know; this isn't covered in your documents.",
    "fa": "نمی‌دانم؛ این موضوع در اسناد شما نیامده است.",
}

# Heading used when no language model is available and passages are shown as-is.
RETRIEVAL_ONLY_TITLES = {
    "en": "Retrieval-only mode (no language model). Most relevant passages:",
    "fa": "حالت فقط‌بازیابی (بدون مدل زبانی). مرتبط‌ترین بخش‌ها:",
}


def no_answer_message(language: str) -> str:
    return NO_ANSWER_MESSAGES["fa"] if language == "fa" else NO_ANSWER_MESSAGES["en"]


def icon_for(path: Path) -> str:
    return EXT_ICONS.get(path.suffix.lower(), "📄")
