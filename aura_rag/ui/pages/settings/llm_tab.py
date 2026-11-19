"""LLM settings tab."""

import json
import os
import streamlit as st

from ....constants import LLM_BACKEND_MODES, OPENAI_PROVIDER_PRESETS
from ....dependencies import OpenAIModule
from ....llm_manager import LLMManager
from ....network import is_offline_mode
from ....rag_application import RAGApplication
from ...components import api_status_label


def render_llm_tab(app: RAGApplication):
    """LLM settings tab."""
    st.caption(
        "💡 Answer speed on a local model is dominated by three things: **model size** "
        "(a 7B model is much slower per token than a 3B one), **GPU Layers** (0 = pure "
        "CPU; if you have a GPU, raising this offloads work to it and is usually the "
        "single biggest speedup available), and **Context Length / Max Tokens** (larger "
        "values mean more work per answer). If generation feels slow, try a smaller "
        "quantized model or raising GPU Layers before anything else."
    )
    st.markdown("##### 🖥️ Offline GGUF model")
    model_path = st.text_input(
        "Model Path (GGUF format)",
        value=app.config.config["llm_model_path"],
        help="Path to your local GGUF model file",
    )

    col1, col2 = st.columns(2)

    with col1:
        context_length = st.number_input(
            "Context Length",
            min_value=512,
            max_value=32768,
            value=app.config.config["llm_context_length"],
            step=512,
        )

        max_tokens = st.number_input(
            "Max Tokens",
            min_value=64,
            max_value=4096,
            value=app.config.config["llm_max_tokens"],
            step=64,
        )

    with col2:
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=app.config.config["llm_temperature"],
            step=0.1,
        )

        gpu_layers = st.number_input(
            "GPU Layers (0 for CPU only)",
            min_value=0,
            max_value=100,
            value=app.config.config["llm_gpu_layers"],
            step=1,
        )

    c1, c2 = st.columns(2)
    with c1:
        if st.button("💾 Save LLM Settings", use_container_width=True):
            app.config.config["llm_model_path"] = model_path
            app.config.config["llm_context_length"] = context_length
            app.config.config["llm_max_tokens"] = max_tokens
            app.config.config["llm_temperature"] = temperature
            app.config.config["llm_gpu_layers"] = gpu_layers
            app.config.save_config()
            st.success("Settings saved! Reload the model for changes to take effect.")
    with c2:
        if st.button("🚀 Load Model", type="primary", use_container_width=True):
            with st.spinner("Loading LLM... this may take a few minutes..."):
                if app.initialize_llm(model_path):
                    st.session_state.llm_loaded = True
                    st.success("✅ LLM loaded successfully!")
                else:
                    st.error("❌ Failed to load LLM. Check the GGUF model path or the OpenAI API key.")

    st.markdown("---")
    st.markdown("##### 📂 Local GGUF models")
    st.caption(
        f"The app doesn't download GGUF models itself. Copy your `.gguf` files into "
        f"`{app.config.models_dir}` (or type any path above)."
    )
    found_ggufs = sorted(p.name for p in app.config.models_dir.glob("*.gguf"))
    if found_ggufs:
        gc1, gc2 = st.columns([3, 1])
        with gc1:
            picked_gguf = st.selectbox("Found in the models folder", found_ggufs, key="found_gguf_select")
        with gc2:
            st.write("")
            if st.button("Use this file", use_container_width=True, key="use_found_gguf_btn"):
                app.config.config["llm_model_path"] = str(app.config.models_dir / picked_gguf)
                app.config.save_config()
                st.rerun()
    else:
        st.caption("No .gguf files found in the models folder yet.")

    # ------------------------------------------------------------------
    # OpenAI / OpenAI-compatible API backend — sits behind the offline
    # GGUF model. Works with OpenAI itself and with any server that
    # speaks the OpenAI chat-completions protocol.
    # ------------------------------------------------------------------
    st.markdown("---")
    st.markdown("##### ☁️ OpenAI / OpenAI-compatible API (optional — sits behind the offline GGUF model)")
    if OpenAIModule is None:
        st.info("Install the SDK to enable this: `pip install openai --break-system-packages`")

    cfg = app.config.config
    saved_key = cfg.get("openai_api_key", "")
    env_key = os.environ.get("OPENAI_API_KEY", "")

    backend_keys = list(LLM_BACKEND_MODES.keys())
    current_mode = cfg.get("llm_backend", "gguf_only")
    backend_mode = st.selectbox(
        "LLM backend",
        options=backend_keys,
        format_func=lambda k: LLM_BACKEND_MODES[k],
        index=backend_keys.index(current_mode) if current_mode in backend_keys else 0,
        key="llm_backend_select",
        help="GGUF only: never uses the API. GGUF first: the API is used only if the "
             "GGUF model isn't loaded or fails before producing any output. "
             "API only: always uses the API.",
    )

    # Offline mode hides the remote providers; local servers stay available.
    preset_keys = [
        k for k, v in OPENAI_PROVIDER_PRESETS.items()
        if not is_offline_mode() or k == "custom" or LLMManager.is_local_url(v["base_url"])
    ]
    current_provider = cfg.get("openai_provider", "openai")
    if current_provider not in preset_keys:
        current_provider = "custom"
    provider = st.selectbox(
        "Provider preset",
        options=preset_keys,
        format_func=lambda k: OPENAI_PROVIDER_PRESETS[k]["label"],
        index=preset_keys.index(current_provider),
        key="openai_provider_select",
        help="Fills in a typical base URL and model. Any OpenAI-compatible server works — "
             "pick 'Custom' and enter your own URL.",
    )
    preset = OPENAI_PROVIDER_PRESETS[provider]
    same_provider = provider == current_provider

    # Widget keys include the provider, so switching presets resets the
    # fields to that preset's defaults instead of keeping stale values.
    openai_base_url = st.text_input(
        "Base URL",
        value=cfg.get("openai_base_url", "") if same_provider else preset["base_url"],
        placeholder="https://api.openai.com/v1",
        help="Include the /v1 part, e.g. http://localhost:1234/v1. Leave empty for api.openai.com.",
        key=f"openai_base_url_input_{provider}",
    )

    endpoint_blocked = is_offline_mode() and not LLMManager.is_local_url(openai_base_url)
    if endpoint_blocked:
        st.error(
            "🔒 Offline mode is on: only local / LAN endpoints are allowed "
            "(e.g. http://localhost:11434/v1 for Ollama). Enter a local URL, or switch to "
            "Online mode in the sidebar to use a remote API."
        )
    elif LLMManager.is_local_url(openai_base_url):
        st.info("🖥️ This endpoint looks local / on your LAN — documents stay on your own network.")
    else:
        st.warning(
            "This is a remote service: your question **and the retrieved document excerpts** "
            "are sent to the provider when it answers. Keep **Offline GGUF only** selected "
            "if your documents must never leave this machine.",
            icon="⚠️",
        )

    if saved_key:
        st.caption(f"🔑 A key is saved in config (…{saved_key[-4:]}).")
    elif env_key:
        st.caption(f"🔑 Using the OPENAI_API_KEY environment variable (…{env_key[-4:]}).")
    elif preset["key_required"]:
        st.caption("No key set yet — this provider requires one.")
    else:
        st.caption("No key set. That's fine for most local/self-hosted servers (a placeholder is sent automatically).")

    key_input = st.text_input(
        "API key",
        type="password",
        value="",
        placeholder="leave blank to keep the saved key" if saved_key else (
            "sk-…" if preset["key_required"] else "optional for local servers"
        ),
        key="openai_key_input",
    )

    default_model = cfg.get("openai_model", "gpt-4o-mini") if same_provider else preset["model"]
    fetched_models = st.session_state.get("openai_fetched_models", {}).get(provider)
    if fetched_models:
        model_options = fetched_models if default_model in fetched_models else ([default_model] + fetched_models if default_model else fetched_models)
        openai_model = st.selectbox(
            f"Model ({len(fetched_models)} available on this server)",
            options=model_options,
            index=model_options.index(default_model) if default_model in model_options else 0,
            key=f"openai_model_select_{provider}",
        )
    else:
        openai_model = st.text_input(
            "Model",
            value=default_model,
            help="Use 'Fetch models' below to list what the server offers.",
            key=f"openai_model_input_{provider}",
        )

    with st.expander("Advanced"):
        extra_headers_text = st.text_area(
            "Extra HTTP headers (JSON object, optional)",
            value=cfg.get("openai_extra_headers", "") if same_provider else "",
            placeholder='{"HTTP-Referer": "https://example.com", "X-Title": "My App"}',
            height=80,
            key=f"openai_headers_input_{provider}",
        )
        use_streaming = st.toggle(
            "Stream tokens",
            value=bool(cfg.get("openai_use_streaming", True)),
            help="Turn off for servers without streaming support. Even when on, the app "
                 "automatically retries without streaming if streaming fails before any text arrives.",
            key="openai_streaming_toggle",
        )

    def _parse_headers(text):
        text = (text or "").strip()
        if not text:
            return {}, None
        try:
            data = json.loads(text)
        except Exception as e:
            return None, f"not valid JSON ({e})"
        if not isinstance(data, dict):
            return None, "must be a JSON object like {\"Header\": \"value\"}"
        return {str(k): str(v) for k, v in data.items()}, None

    parsed_headers, header_error = _parse_headers(extra_headers_text)
    if header_error:
        st.error(f"Extra headers: {header_error}")

    def _client_from_form():
        return app.llm_manager.build_openai_client(
            api_key=key_input.strip() or None,
            base_url=openai_base_url.strip(),
            extra_headers=parsed_headers,
        )

    bc1, bc2, bc3, bc4 = st.columns(4)
    with bc1:
        if st.button("💾 Save", use_container_width=True, key="save_openai_btn", disabled=bool(header_error) or endpoint_blocked):
            cfg["llm_backend"] = backend_mode
            cfg["openai_provider"] = provider
            cfg["openai_base_url"] = openai_base_url.strip()
            if key_input.strip():
                cfg["openai_api_key"] = key_input.strip()
            cfg["openai_model"] = (openai_model or "").strip() or preset["model"] or "gpt-4o-mini"
            cfg["openai_extra_headers"] = extra_headers_text.strip()
            cfg["openai_use_streaming"] = bool(use_streaming)
            app.config.save_config()
            if backend_mode in ("gguf_then_openai", "openai_only"):
                app.llm_manager.initialize_openai()
            st.session_state.llm_loaded = app.llm_manager.is_ready()
            st.success(
                "Saved. " + (
                    "Backend ready." if st.session_state.llm_loaded
                    else "Backend not ready yet — load the GGUF model or check the endpoint settings."
                )
            )
    with bc2:
        if st.button("📋 Fetch models", use_container_width=True, key="fetch_models_btn", disabled=bool(header_error) or endpoint_blocked):
            try:
                client = _client_from_form()
                ids = sorted({m.id for m in client.models.list()})
                if not ids:
                    st.warning("The server returned an empty model list. Type the model name manually.")
                else:
                    st.session_state.setdefault("openai_fetched_models", {})[provider] = ids
                    st.rerun()
            except Exception as e:
                st.error(f"❌ Could not list models ({type(e).__name__}: {e}). "
                         f"Some servers don't expose /models — type the model name manually.")
    with bc3:
        if st.button("🧪 Test", use_container_width=True, key="test_openai_btn", disabled=bool(header_error) or endpoint_blocked):
            try:
                client = _client_from_form()
                test_kwargs = dict(
                    model=(openai_model or "").strip() or preset["model"] or "gpt-4o-mini",
                    messages=[{"role": "user", "content": "Reply with the single word: ok"}],
                )
                try:
                    resp = client.chat.completions.create(max_tokens=16, **test_kwargs)
                except Exception:
                    resp = client.chat.completions.create(max_completion_tokens=256, **test_kwargs)
                reply = (resp.choices[0].message.content or "").strip()
                app.llm_manager.api_verified = True
                app.llm_manager.api_verify_error = ""
                st.success(f"✅ Endpoint works — model replied: {reply!r}")
            except Exception as e:
                app.llm_manager.api_verified = False
                app.llm_manager.api_verify_error = f"{type(e).__name__}: {e}"
                st.error(f"❌ {type(e).__name__}: {e}")
    with bc4:
        if st.button(
            "🗑️ Remove key", use_container_width=True,
            disabled=not saved_key, key="remove_openai_key_btn",
        ):
            cfg["openai_api_key"] = ""
            app.config.save_config()
            app.llm_manager.openai_client = None
            app.llm_manager.api_verified = None
            st.session_state.llm_loaded = app.llm_manager.is_ready()
            st.rerun()

    # ---- API connection status badge --------------------------------
    api_code, api_detail = app.llm_manager.api_status()
    badge = f"**API status:** {api_status_label(app)}"
    if api_detail and api_code in ("connected", "ready"):
        badge += f" — `{api_detail}`"
    if api_code == "connected":
        st.success(badge)
    elif api_code in ("no_key", "failed", "blocked"):
        st.error(badge + (f"\n\n{api_detail}" if api_code == "failed" and api_detail else ""))
    elif api_code in ("ready", "not_ready"):
        st.warning(badge + ("  \nPress 🧪 Test to verify it." if api_code == "ready" else ""))
    else:
        st.info(badge + "  \nSwitch the LLM backend above to enable it.")
