"""Retrieval settings tab."""

import streamlit as st

from ....constants import PRESETS
from ....dependencies import BM25Retriever, CrossEncoder, EnsembleRetriever
from ....rag_application import RAGApplication


def render_retrieval_tab(app: RAGApplication):
    """Retrieval settings tab."""
    st.caption(f"Currently active: **{app.retriever.describe_active_pipeline()}**")

    st.markdown("##### ⚡ Presets")
    _pcols = st.columns(len(PRESETS))
    for _col, _name in zip(_pcols, PRESETS):
        if _col.button(_name, use_container_width=True, key=f"preset_{_name}"):
            app.config.config.update(PRESETS[_name])
            app.config.save_config()
            st.rerun()
    st.caption("Fast = fewer, shorter passages · Balanced · Thorough = more evidence, slower. "
               "Sets candidates, final chunks, neighbour chunks and the evidence budget.")

    st.markdown("##### 🔀 Hybrid search")
    hybrid_available = BM25Retriever is not None and EnsembleRetriever is not None
    if not hybrid_available:
        st.info(
            "Hybrid keyword search isn't available — install `rank_bm25` "
            "(and `langchain-classic` or `langchain`) to enable it. See the "
            "import banner at the top of the app for exact commands."
        )
    use_hybrid = st.toggle(
        "Blend BM25 keyword search with vector search",
        value=app.config.config.get("use_hybrid_search", True) and hybrid_available,
        disabled=not hybrid_available,
        help="Catches exact terms, names, codes, and acronyms that pure semantic "
             "similarity can miss. Recommended when your documents mix scripts "
             "(e.g. Persian text with English product codes or numbers).",
    )
    hybrid_weight = st.slider(
        "Vector weight (remainder goes to BM25)",
        min_value=0.0, max_value=1.0,
        value=float(app.config.config.get("hybrid_vector_weight", 0.6)),
        step=0.05,
        disabled=not (use_hybrid and hybrid_available),
    )

    st.markdown("##### 🧩 Multi-query expansion")
    multi_query = st.toggle(
        "Ask the LLM to rephrase the question before retrieving",
        value=app.config.config.get("use_multi_query", False),
        disabled=not st.session_state.get("llm_loaded", False),
        help="Retrieves for several phrasings of the same question and merges the "
             "results — helps when the wording of a question doesn't match the "
             "wording in the documents. Requires the LLM to be loaded.",
    )
    if not st.session_state.get("llm_loaded", False):
        st.caption("Load the LLM (in the LLM tab) to enable multi-query expansion.")
    if multi_query:
        st.warning(
            "⚠️ This runs a full extra LLM generation *before* retrieval even starts, "
            "on top of the answer generation itself — with a local CPU model this can "
            "roughly double how long each answer takes (with OpenAI it costs one extra "
            "API call per question). If answers feel slow, try turning this off first.",
            icon="⚠️",
        )
    multi_query_count = st.number_input(
        "Number of alternate phrasings to generate",
        min_value=1, max_value=6,
        value=int(app.config.config.get("multi_query_count", 3)),
        disabled=not multi_query,
        help="Each additional phrasing means more retrieval work per question, but "
             "does not add another LLM call (only the rephrasing step above does).",
    )

    st.markdown("##### 🎯 Cross-encoder reranking")
    reranker_available = CrossEncoder is not None
    if not reranker_available:
        st.info(
            "Reranking isn't available — install `sentence-transformers` to "
            "enable it. See the import banner at the top of the app."
        )
    use_reranker = st.toggle(
        "Rerank candidates with a cross-encoder before answering",
        value=app.config.config.get("use_reranker", True) and reranker_available,
        disabled=not reranker_available,
        help="Scores every retrieved chunk directly against the question with a "
             "small, more precise model, and keeps only the best ones.",
    )
    reranker_model = st.text_input(
        "Reranker model (HuggingFace cross-encoder)",
        value=app.config.config.get("reranker_model", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
        disabled=not use_reranker,
        help="For multilingual/Persian corpora, consider a multilingual cross-encoder "
             "such as 'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1'. Offline: enter a local "
             "folder path, or copy the model folder into the app's models/rerankers folder.",
    )
    rerank_min_score = st.number_input(
        "Relevance threshold (chunks scoring below this are dropped)",
        min_value=-20.0, max_value=20.0,
        value=float(app.config.config.get("rerank_min_score", -4.0)),
        step=0.5,
        disabled=not (use_reranker and reranker_available),
        help="Lets the assistant say \"I don't know\" for questions that aren't about "
             "your documents: if no chunk reaches this score, the model isn't called at all. "
             "Start at -4 for ms-marco-MiniLM-L-6-v2. Raise it (e.g. -2 or 0) if unrelated "
             "questions still get answers; lower it if relevant questions are being "
             "rejected. A different reranker model uses a different score scale, so "
             "re-tune it after switching models.",
    )

    max_vector_distance = st.number_input(
        "Max vector distance (farther chunks are dropped; 0 = off)",
        min_value=0.0, max_value=2.0,
        value=float(app.config.config.get("max_vector_distance", 1.1)),
        step=0.05,
        help="Works even without a reranker. Smaller = stricter. If unrelated questions "
             "still get answers, lower it (e.g. 0.9); if relevant questions are rejected, "
             "raise it (e.g. 1.3). Keyword hits must also share enough meaningful words "
             "with the question.",
    )

    require_term_overlap = st.toggle(
        "Require the question's key words to appear in the retrieved text",
        value=bool(app.config.config.get("require_term_overlap", True)),
        help="Model-independent safety net: a question like 'what is the capital of "
             "France' is rejected (\"I don't know\") when its meaningful words don't "
             "appear in any retrieved chunk. Turn off if you ask questions in a different "
             "language than your documents, or paraphrase heavily.",
    )
    show_retrieval_debug = st.toggle(
        "Show a retrieval debug panel under each answer",
        value=bool(app.config.config.get("show_retrieval_debug", True)),
        help="Shows vector distances, rerank scores and which gate rejected a question — "
             "use it to tune the thresholds above.",
    )

    C = app.config.config
    st.markdown("##### 🕸️ Entity-graph search")
    use_graph = st.toggle(
        "Add entity-graph search to the fusion",
        value=bool(C.get("use_graph_search", True)),
        help="Finds chunks that mention the same named entities (people, products, organisations) "
             "as the question, weighted by how rare the entity is. Works on Latin-script names.",
    )
    graph_weight = st.slider("Graph weight in fusion", 0.0, 1.0, float(C.get("graph_weight", 0.35)), 0.05,
                             disabled=not use_graph)
    st.caption(f"{app.persistent_store.entity_count():,} entity mentions indexed. Documents indexed "
               f"before the entity graph existed need a one-time rebuild.")
    if st.button("🔁 Rebuild entity index", key="rebuild_entities_btn"):
        with st.spinner("Extracting entities..."):
            _n = app.persistent_store.rebuild_entities()
        st.success(f"Entity index rebuilt from {_n:,} chunks.")

    st.markdown("##### 🧭 Query understanding")
    use_condense = st.toggle(
        "Rewrite follow-up questions into standalone search queries",
        value=bool(C.get("use_condense", True)),
        help="'What about its price?' is searched as a full question using the conversation. "
             "Uses the LLM when loaded, otherwise a simple heuristic.",
    )
    use_hyde = st.toggle(
        "HyDE: also search with a hypothetical answer passage",
        value=bool(C.get("use_hyde", False)),
        disabled=not st.session_state.get("llm_loaded", False),
        help="Costs one extra LLM generation per question. Needs the LLM loaded.",
    )

    st.markdown("##### 🧩 Evidence assembly")
    use_mmr = st.toggle(
        "Diversify results (MMR) to avoid near-duplicate passages",
        value=bool(C.get("use_mmr", True)),
    )
    mmr_redundancy = st.slider("Diversity penalty", 0.0, 0.8, float(C.get("mmr_redundancy", 0.3)), 0.05,
                               disabled=not use_mmr)
    neighbors = st.slider("Neighbour chunks per side", 0, 3, int(C.get("neighbors", 1)),
                          help="Adds the chunks before/after each hit and merges overlapping spans into one passage.")
    max_context_chars = st.slider("Evidence budget (characters)", 2000, 40000,
                                  int(C.get("max_context_chars", 6000)), 500,
                                  help="Keep this within your model's context window.")

    st.markdown("##### ✅ Answer verification")
    verify_answers = st.toggle(
        "Check every answer sentence against its cited sources",
        value=bool(C.get("verify_answers", True)),
        help="Embeds each answer sentence and compares it with the passages: flags unsupported, "
             "mis-cited and uncited claims.",
    )
    verify_threshold = st.slider("Verification threshold", 0.3, 0.8, float(C.get("verify_threshold", 0.45)),
                                 0.01, disabled=not verify_answers,
                                 help="Cosine similarity above which a sentence counts as supported.")

    col1, col2 = st.columns(2)
    with col1:
        rerank_candidates = st.number_input(
            "Candidates fetched per query",
            min_value=5, max_value=100,
            value=int(app.config.config.get("rerank_candidates", 20)),
            step=5,
            help="How many chunks each retrieval stage pulls before merging/reranking.",
        )
    with col2:
        final_k = st.number_input(
            "Final chunks used to answer",
            min_value=1, max_value=20,
            value=int(app.config.config.get("final_k", 5)),
        )

    if st.button("💾 Save Retrieval Settings", type="primary", use_container_width=True):
        app.config.config["use_hybrid_search"] = use_hybrid and hybrid_available
        app.config.config["hybrid_vector_weight"] = hybrid_weight
        app.config.config["use_multi_query"] = multi_query
        app.config.config["multi_query_count"] = multi_query_count
        app.config.config["use_reranker"] = use_reranker and reranker_available
        app.config.config["reranker_model"] = reranker_model
        app.config.config["rerank_min_score"] = float(rerank_min_score)
        app.config.config["max_vector_distance"] = float(max_vector_distance)
        app.config.config["require_term_overlap"] = bool(require_term_overlap)
        app.config.config["show_retrieval_debug"] = bool(show_retrieval_debug)
        app.config.config["rerank_candidates"] = rerank_candidates
        app.config.config["final_k"] = final_k
        app.config.config.update({
            "use_graph_search": bool(use_graph), "graph_weight": float(graph_weight),
            "use_condense": bool(use_condense), "use_hyde": bool(use_hyde),
            "use_mmr": bool(use_mmr), "mmr_redundancy": float(mmr_redundancy),
            "neighbors": int(neighbors), "max_context_chars": int(max_context_chars),
            "verify_answers": bool(verify_answers), "verify_threshold": float(verify_threshold),
        })
        app.config.save_config()
        st.success(f"Saved. Active pipeline: {app.retriever.describe_active_pipeline()}")
