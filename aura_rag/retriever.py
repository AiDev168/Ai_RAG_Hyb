"""Hybrid retrieval: vector + FTS5/BM25 + entity graph, reranking, MMR, neighbours."""

import logging
import re
from typing import List, Optional

from .config import Config, resolve_local_model
from .dependencies import CrossEncoder, Document
from .persistent_store import PersistentStore
from .text_utils import (
    HYDE_PROMPT,
    _join_overlapping,
    _normalize_search_text,
    extract_entities,
    page_label_from,
)
from .vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


class AdvancedRetriever:
    """Configurable, multi-stage retrieval pipeline built on top of the
    vector store:

      1. (optional) Multi-query expansion — asks the LLM for alternate
         phrasings of the question so retrieval isn't at the mercy of a
         single wording.
      2. Hybrid fetch — for each query variant, blend dense vector search
         with sparse BM25 keyword search (EnsembleRetriever), so exact
         terms/names/codes that embeddings can miss still surface.
      3. Merge + dedupe candidates across all query variants.
      4. (optional) Cross-encoder reranking — a small model scores every
         candidate directly against the *original* question, drops chunks
         scoring below "rerank_min_score" (so unrelated questions end up
         with NO chunks and get the fixed "I don't know" reply), and keeps
         only the top few.

    Every stage degrades gracefully: if a dependency isn't installed or a
    stage is toggled off, the pipeline just falls through to plain vector
    search on the original query.

    All stages accept an optional file_names list that restricts retrieval
    to chunks from those documents only (None = all documents).
    """

    def __init__(self, config: Config, vector_manager: "VectorStoreManager", chunk_store: PersistentStore):
        self.config = config
        self.vector_manager = vector_manager
        self.chunk_store = chunk_store
        self._cross_encoder = None
        self._cross_encoder_name = None
        # Filled on every retrieve() so the chat page can show WHY a question
        # was answered or rejected (distances, rerank scores, term-overlap gate).
        self.last_debug: dict = {}

    def mark_dirty(self) -> None:
        # No-op with the SQLite/FTS5-backed store: search_keyword() always
        # reads the current on-disk index directly, so there is nothing to
        # rebuild or invalidate. Kept as a method so callers written against
        # the old in-memory-index API don't need to change.
        pass

    def _get_cross_encoder(self):
        if CrossEncoder is None:
            return None
        model_name = self.config.config.get("reranker_model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        if self._cross_encoder is None or self._cross_encoder_name != model_name:
            try:
                self._cross_encoder = CrossEncoder(
                    resolve_local_model(model_name, self.config.models_dir, "rerankers")
                )
                self._cross_encoder_name = model_name
            except Exception:
                logger.exception(
                    f"Failed to load cross-encoder '{model_name}' — reranking is skipped. "
                    f"Offline: put the model folder in {self.config.models_dir / 'rerankers'}."
                )
                self._cross_encoder = None
        return self._cross_encoder

    @staticmethod
    def _doc_key(doc) -> tuple:
        return (doc.metadata.get("file_name", ""), hash(doc.page_content))

    _STOPWORDS = {
        "what", "which", "who", "whom", "whose", "when", "where", "why", "how",
        "is", "are", "was", "were", "the", "and", "for", "that", "this", "with",
        "from", "does", "did", "can", "about", "tell", "please", "give",
        "چه", "چی", "کی", "کجا", "چرا", "چگونه", "چطور", "است", "هست", "این",
        "آن", "که", "را", "از", "به", "در", "با", "برای", "یا", "و",
    }

    _STOPWORDS = _STOPWORDS | {
        "have", "has", "you", "your", "not", "any", "there", "their", "them",
        "into", "some", "than", "then", "also", "explain", "describe", "list",
        "mention", "give", "show", "know", "need", "want", "would", "could",
        "should", "will", "just", "like", "more", "most", "many", "much",
        "های", "ها", "یک", "هم", "ولی", "اما", "اگر", "من", "تو", "ما",
        "شما", "آیا", "بگو", "بگویید", "توضیح", "بده", "بدهید",
    }
    # Questions like "summarize my documents" legitimately share no words with
    # the documents, so they skip the term-overlap gate.
    _GENERIC_INTENT = {"summarize", "summarise", "summary", "overview", "خلاصه", "مرور", "خلاصه‌ی"}

    def _term_overlap_ok(self, query: str, docs: List):
        """Model-independent grounding gate. The question's meaningful words
        (stopwords removed, light prefix-stemming) must appear in the retrieved
        text: all of them for 1-2 word questions, otherwise at least
        ceil(ratio * n). 'What is the capital of France' -> {capital, france};
        neither appears in unrelated documents, so the question is rejected
        without calling the LLM. Returns (ok, explanation)."""
        import math
        q = _normalize_search_text(query).lower()
        all_words = re.findall(r"\w+", q, flags=re.UNICODE)
        if any(w in self._GENERIC_INTENT for w in all_words):
            return True, "bypassed (generic summary-style question)"
        terms = sorted({w for w in all_words if len(w) >= 3 and w not in self._STOPWORDS})
        if not terms:
            return True, "no meaningful terms to check"
        text = " ".join(_normalize_search_text(d.page_content).lower() for d in docs)

        def present(w: str) -> bool:
            stem = w if len(w) <= 4 else (w[:-1] if len(w) <= 5 else w[:-2])
            return stem in text

        hits = [w for w in terms if present(w)]
        n = len(terms)
        ratio = float(self.config.config.get("term_overlap_ratio", 0.5))
        needed = n if n <= 2 else math.ceil(ratio * n)
        missing = [w for w in terms if w not in hits]
        ok = len(hits) >= needed
        return ok, f"{len(hits)}/{n} question terms found in context (need {needed}); missing: {missing}"

    def _keyword_relevant(self, query: str, doc) -> bool:
        """A keyword hit only counts if at least half of the question's
        meaningful words (stopwords removed) appear in the chunk. This stops
        'what is the capital of France' from matching any chunk containing 'is'."""
        q = _normalize_search_text(query).lower()
        words = {w for w in re.findall(r"\w+", q, flags=re.UNICODE)
                 if len(w) >= 3 and w not in self._STOPWORDS}
        if not words:
            return False
        text = _normalize_search_text(doc.page_content).lower()
        hits = sum(1 for w in words if w in text)
        return hits / len(words) >= 0.5

    def _fetch_candidates(self, query: str, k: int, file_names: Optional[List[str]] = None,
                          vector_only: bool = False) -> List:
        # Vector hits farther than max_vector_distance are dropped (embeddings
        # are normalized, so this is a similarity floor); keyword hits must
        # share enough meaningful words with the question. Unrelated questions
        # therefore end up with NO candidates -> fixed "I don't know" reply.
        max_dist = float(self.config.config.get("max_vector_distance", 1.1))

        vector_docs: List = []
        vs = self.vector_manager.vectorstore
        if vs is not None:
            try:
                kwargs = {"k": k}
                if file_names:
                    kwargs["filter"] = {"file_name": {"$in": list(file_names)}}
                pairs = vs.similarity_search_with_score(query, **kwargs)
                dbg = self.last_debug
                dbg.setdefault("vector_best", []).extend(
                    (d.metadata.get("file_name", ""), round(float(dist), 3)) for d, dist in pairs[:5]
                )
                vector_docs = [d for d, dist in pairs if max_dist <= 0 or dist <= max_dist]
            except Exception:
                logger.exception("Vector retrieval failed")
                vector_docs = []

        keyword_docs: List = []
        if not vector_only and self.config.config.get("use_hybrid_search", True):
            raw = self.chunk_store.search_keyword(query, k=k, file_names=file_names)
            keyword_docs = [d for d in raw if self._keyword_relevant(query, d)]
            self.last_debug["keyword_raw"] = self.last_debug.get("keyword_raw", 0) + len(raw)
            self.last_debug["keyword_kept"] = self.last_debug.get("keyword_kept", 0) + len(keyword_docs)

        entity_docs: List = []
        if not vector_only and self.config.config.get("use_graph_search", True):
            try:
                entity_docs = self.chunk_store.search_entities(query, k=k, file_names=file_names)
            except Exception:
                logger.exception("Entity-graph search failed")
            self.last_debug["graph_hits"] = self.last_debug.get("graph_hits", 0) + len(entity_docs)

        w_vec, w_kw, w_graph = self._adaptive_weights(query)
        non_empty = [(lst, w) for lst, w in ((vector_docs, w_vec), (keyword_docs, w_kw), (entity_docs, w_graph)) if lst]
        if not non_empty:
            return []
        if len(non_empty) == 1:
            return non_empty[0][0]

        # Weighted reciprocal rank fusion of the (already filtered) ranked lists.
        rank_constant = 60
        scores: dict = {}
        doc_by_key: dict = {}
        for lst, weight in non_empty:
            for rank, doc in enumerate(lst):
                key = self._doc_key(doc)
                doc_by_key[key] = doc
                scores[key] = scores.get(key, 0.0) + weight * (1.0 / (rank_constant + rank + 1))
        ordered = sorted(scores, key=lambda key: scores[key], reverse=True)
        return [doc_by_key[key] for key in ordered[:k]]

    def _expand_queries(self, query: str, llm) -> List[str]:
        if not self.config.config.get("use_multi_query", False) or llm is None:
            return [query]

        n = int(self.config.config.get("multi_query_count", 3))
        expansion_prompt = (
            f"Generate {n} different ways to phrase the following question, each "
            f"capturing the same intent but with different wording or emphasis. "
            f"Return exactly {n} lines, one question per line, with no numbering, "
            f"bullets, or extra commentary. Respond in the same language as the "
            f"question.\n\nQuestion: {query}"
        )
        try:
            raw = llm.invoke(expansion_prompt)
            text = raw if isinstance(raw, str) else str(raw)
            variants = [line.strip(" \t-•*") for line in text.splitlines() if line.strip()]
            variants = [v for v in variants if v and v.lower() != query.lower()][:n]
            return [query] + variants
        except Exception:
            logger.exception("Multi-query expansion failed, falling back to the original query only")
            return [query]

    def _rerank(self, query: str, documents: List, top_k: int) -> List:
        """Score every candidate against the question with the cross-encoder,
        drop the ones below "rerank_min_score" (irrelevant chunks), and keep
        the best top_k. If nothing clears the threshold the result is empty,
        which makes the caller answer with the fixed "I don't know" message
        without ever calling the LLM."""
        cross_encoder = self._get_cross_encoder()
        if cross_encoder is None:
            logger.warning("Reranker NOT loaded - relevance threshold is inactive; relying on vector/keyword gates only.")
            self.last_debug["reranker_loaded"] = False
            return documents[:top_k]
        if not documents:
            return []
        try:
            pairs = [[query, doc.page_content] for doc in documents]
            scores = cross_encoder.predict(pairs)
            min_score = float(self.config.config.get("rerank_min_score", -4.0))
            scored = sorted(zip(scores, documents), key=lambda pair: pair[0], reverse=True)
            kept = []
            for score, doc in scored:
                if float(score) >= min_score:
                    doc.metadata["_score"] = float(score)
                    kept.append(doc)
            self.last_debug["reranker_loaded"] = True
            self.last_debug["rerank_top_scores"] = [round(float(sc), 2) for sc, _ in scored[:6]]
            self.last_debug["rerank_min_score"] = min_score
            return kept[:top_k]
        except Exception:
            logger.exception("Cross-encoder reranking failed, returning unranked candidates")
            return documents[:top_k]

    def retrieve(self, query: str, llm=None, file_names: Optional[List[str]] = None) -> List:
        candidate_k = int(self.config.config.get("rerank_candidates", 20))
        final_k = int(self.config.config.get("final_k", 5))

        cfg = self.config.config
        queries = self._expand_queries(query, llm)
        self.last_debug = {"queries": queries, "candidates_before_rerank": 0}

        seen = set()
        merged: List = []

        def _add(found):
            for doc in found:
                key = (doc.metadata.get("file_name", ""), hash(doc.page_content))
                if key not in seen:
                    seen.add(key)
                    merged.append(doc)

        for q in queries:
            _add(self._fetch_candidates(q, k=candidate_k, file_names=file_names))

        hyde = self._hyde(query, llm) if (llm is not None and cfg.get("use_hyde", False)) else None
        if hyde:
            self.last_debug["hyde"] = hyde
            _add(self._fetch_candidates(hyde, k=candidate_k, file_names=file_names, vector_only=True))

        self.last_debug["candidates_before_rerank"] = len(merged)
        pool_k = final_k * 3 if cfg.get("use_mmr", True) else final_k
        if cfg.get("use_reranker", True):
            docs = self._rerank(query, merged, pool_k)
        else:
            self.last_debug["reranker_loaded"] = "disabled in settings"
            docs = merged[:pool_k]
        for rank, doc in enumerate(docs):
            doc.metadata.setdefault("_score", 1.0 / (rank + 1))
        docs = self._mmr(docs, final_k)
        self.last_debug["after_rerank"] = len(docs)

        if docs and self.config.config.get("require_term_overlap", True):
            ok, info = self._term_overlap_ok(query, docs)
            self.last_debug["term_gate"] = info
            if not ok:
                docs = []
        self.last_debug["final_chunks"] = len(docs)
        return docs

    def _adaptive_weights(self, query: str):
        """(vector, keyword, graph) weights for the fusion, nudged by the kind of question."""
        cfg = self.config.config
        wv = min(max(float(cfg.get("hybrid_vector_weight", 0.6)), 0.0), 1.0)
        wk = 1.0 - wv
        wg = float(cfg.get("graph_weight", 0.35)) if cfg.get("use_graph_search", True) else 0.0
        q = query.lower()
        if re.search(r"\b(exactly|quote|definition|number|date|table|version|equation|section)\b", q):
            wk *= 1.4
        if re.search(r"\b(relationship|related|connection|compare|between|why|how does|impact)\b", q):
            wg *= 2.0
        if len(extract_entities(query)) >= 2:
            wg *= 1.5
            wk *= 1.2
        return wv, wk, wg

    def _hyde(self, query: str, llm) -> Optional[str]:
        try:
            out = llm.invoke(f"{HYDE_PROMPT}\n\nQuestion: {query}\n\nPassage:")
            return out if len(out) > 20 else None
        except Exception:
            logger.exception("HyDE generation failed")
            return None

    def _mmr(self, docs: List, k: int) -> List:
        """Maximal marginal relevance on embedding similarity: keeps relevant chunks, drops near-duplicates."""
        if len(docs) <= k or not self.config.config.get("use_mmr", True):
            return docs[:k]
        try:
            import numpy as np
            vecs = np.asarray(self.vector_manager.embeddings.embed_documents([d.page_content for d in docs]),
                              dtype="float32")
            vecs = vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-9)
            rel = np.array([float(d.metadata.get("_score", 0.0)) for d in docs], dtype=float)
            span = rel.max() - rel.min()
            rel_n = (rel - rel.min()) / span if span > 0 else np.ones(len(docs))
            sim = vecs @ vecs.T
            lam = float(self.config.config.get("mmr_redundancy", 0.3))
            selected: List[int] = []
            cand = list(range(len(docs)))
            while cand and len(selected) < k:
                best = max(cand, key=lambda i: rel_n[i] - (lam * max(sim[i, j] for j in selected) if selected else 0))
                selected.append(best)
                cand.remove(best)
            return [docs[i] for i in selected]
        except Exception:
            logger.exception("MMR diversification failed; using plain top-k")
            return docs[:k]

    def build_context(self, docs: List) -> List[dict]:
        """Expand each chosen chunk with its neighbours, merge overlapping spans per file,
        enforce the character budget (best spans first) and number the result [1], [2]..."""
        cfg = self.config.config
        neighbors = max(0, int(cfg.get("neighbors", 1)))
        max_chars = int(cfg.get("max_context_chars", 6000))
        store = self.chunk_store
        by_file: dict = {}
        spans: List[dict] = []
        for doc in docs:
            fn = doc.metadata.get("file_name", "")
            score = float(doc.metadata.get("_score", 0.0))
            cid = store.find_chunk_id(fn, doc.page_content) if neighbors else None
            if cid is None:
                page = doc.metadata.get("page")
                spans.append({"file_name": fn, "text": doc.page_content, "score": score,
                              "pages": [page] if isinstance(page, int) else [],
                              "ocr": bool(doc.metadata.get("ocr"))})
            else:
                by_file.setdefault(fn, []).append([cid - neighbors, cid + neighbors, score])

        for fn, ranges in by_file.items():
            ranges.sort(key=lambda r: r[0])
            merged: List[list] = []
            for lo, hi, sc in ranges:
                if merged and lo <= merged[-1][1] + 1:
                    merged[-1][1] = max(merged[-1][1], hi)
                    merged[-1][2] = max(merged[-1][2], sc)
                else:
                    merged.append([lo, hi, sc])
            for lo, hi, sc in merged:
                rows = store.fetch_range(fn, lo, hi)
                if not rows:
                    continue
                pages = [m["page"] for _i, _c, m in rows if isinstance(m.get("page"), int)]
                spans.append({"file_name": fn, "text": _join_overlapping([c for _i, c, _m in rows]),
                              "score": sc, "pages": pages,
                              "ocr": any(m.get("ocr") for _i, _c, m in rows)})

        spans.sort(key=lambda s: s["score"], reverse=True)
        kept: List[dict] = []
        used = 0
        for s in spans:
            room = max_chars - used
            if room <= 0 and kept:
                break
            if len(s["text"]) > room and kept:
                continue
            if len(s["text"]) > max_chars:
                s["text"] = s["text"][:max_chars]
            kept.append(s)
            used += len(s["text"])

        return [{"n": i, "file_name": s["file_name"], "text": s["text"], "score": float(s["score"]),
                 "pages": page_label_from(sorted(set(s["pages"]))), "ocr": s["ocr"]}
                for i, s in enumerate(kept, 1)]

    def describe_active_pipeline(self) -> str:
        stages = ["Vector search"]
        if self.config.config.get("use_hybrid_search", True) and Document is not None:
            stages.append("FTS5 hybrid")
        if self.config.config.get("use_graph_search", True):
            stages.append("entity graph")
        if self.config.config.get("use_multi_query", False):
            stages.append(f"multi-query ×{self.config.config.get('multi_query_count', 3)}")
        if self.config.config.get("use_reranker", True) and CrossEncoder is not None:
            stages.append("cross-encoder rerank")
        if self.config.config.get("use_mmr", True):
            stages.append("MMR")
        return " → ".join(stages)
