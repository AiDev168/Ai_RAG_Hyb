"""RAGApplication: ties config, stores, retriever and LLM together."""

import logging
import re
import time
from pathlib import Path
from typing import List, Optional

from .config import Config
from .constants import PROMPT_VERSION, no_answer_message
from .document_manager import DocumentManager
from .document_processor import DocumentProcessor
from .llm_manager import LLMManager
from .persistent_store import PersistentStore
from .retriever import AdvancedRetriever
from .text_utils import (
    CONDENSE_PROMPT,
    FOLLOWUP_HINT,
    _CITE_RE,
    build_evidence,
    cited_numbers,
    extractive_answer,
    split_sentences,
)
from .vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


class RAGApplication:
    """Main RAG Application class"""

    def __init__(self):
        self.config = Config()
        self.doc_processor = DocumentProcessor(self.config)
        self.vector_manager = VectorStoreManager(self.config)
        self.llm_manager = LLMManager(self.config)
        self.doc_manager = DocumentManager(self.config)
        self.persistent_store = PersistentStore(self.config)
        self.retriever = AdvancedRetriever(self.config, self.vector_manager, self.persistent_store)

        self.vector_manager.load_vectorstore()

    def add_documents(self, uploaded_files, progress_callback=None):
        saved_paths = []

        for uploaded_file in uploaded_files:
            path = self.doc_manager.save_uploaded_file(uploaded_file)
            if path:
                saved_paths.append(path)

        if not saved_paths:
            return False, "No files were saved", []

        try:
            documents, failures = self.doc_processor.load_documents(saved_paths, progress_callback)

            if not documents:
                return False, "No documents could be loaded", failures

            if progress_callback:
                progress_callback(0, 1, "Splitting documents into chunks...")

            chunks = self.doc_processor.split_documents(documents)

            if progress_callback:
                progress_callback(0, 1, f"Creating embeddings for {len(chunks)} chunks...")

            success = self.vector_manager.create_or_update_vectorstore(chunks, progress_callback)

            if success:
                self.persistent_store.append(chunks)
                for p in saved_paths:
                    if p.name not in {f for f, _ in failures}:
                        self.persistent_store.mark_file(p, status="done", chunk_count=0)
                self.retriever.mark_dirty()

            loaded_count = len(saved_paths) - len(failures)
            if success:
                if failures:
                    return True, f"Added {loaded_count} of {len(saved_paths)} files ({len(chunks)} chunks). {len(failures)} file(s) were skipped.", failures
                return True, f"Successfully added {loaded_count} files ({len(chunks)} chunks)", failures
            else:
                return False, "Failed to create vector store (embedding model or Chroma DB error — check the import banner and logs)", failures

        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            return False, str(e), []

    def initialize_llm(self, model_path: str = None):
        return self.llm_manager.initialize_llm(model_path)

    def _corpus_fingerprint(self) -> str:
        """Cheap signature of 'what would change the answer to this exact
        question': the chunk count (changes on add/delete/clear) plus every
        retrieval-affecting setting, plus which LLM backend/model answers,
        plus PROMPT_VERSION (so changing the answering prompt invalidates
        older cached answers). Used to invalidate cached answers without
        needing to hash the whole corpus."""
        cfg = self.config.config
        backend = cfg.get("llm_backend", "gguf_only")
        parts = [
            PROMPT_VERSION,
            str(self.vector_manager.get_document_count()),
            cfg.get("embedding_model", ""),
            str(cfg.get("use_hybrid_search", True)),
            str(cfg.get("hybrid_vector_weight", 0.6)),
            str(cfg.get("use_reranker", True)),
            cfg.get("reranker_model", ""),
            str(cfg.get("rerank_candidates", 20)),
            str(cfg.get("rerank_min_score", -4.0)),
            str(cfg.get("max_vector_distance", 1.1)),
            str(cfg.get("require_term_overlap", True)),
            str(cfg.get("final_k", 5)),
            str(cfg.get("use_multi_query", False)),
            str(cfg.get("multi_query_count", 3)),
            str(cfg.get("use_graph_search", True)),
            str(cfg.get("graph_weight", 0.35)),
            str(cfg.get("use_hyde", False)),
            str(cfg.get("use_mmr", True)),
            str(cfg.get("neighbors", 1)),
            str(cfg.get("max_context_chars", 6000)),
            str(cfg.get("use_condense", True)),
            backend,
            cfg.get("openai_model", "") if backend != "gguf_only" else "",
            cfg.get("openai_base_url", "") if backend != "gguf_only" else "",
        ]
        return "|".join(parts)

    @staticmethod
    def _cache_key(question: str, language: str, file_names: Optional[List[str]] = None) -> str:
        """The document scope is part of the key, so an answer cached for
        'all documents' is never served for a different selection."""
        import hashlib
        normalized = " ".join(question.strip().lower().split())
        scope = "|".join(sorted(file_names)) if file_names else "*"
        return hashlib.sha256(f"{language}::{scope}::{normalized}".encode("utf-8")).hexdigest()

    @staticmethod
    def _build_history_text(chat_history: Optional[List[dict]], max_turns: int) -> str:
        """Render the last max_turns user/assistant exchanges as plain text
        for the prompt. Kept short deliberately — this is context for
        resolving references, not material to answer from, so it shouldn't
        compete with the retrieved document chunks for the model's
        attention or eat into a small context window."""
        if not chat_history or max_turns <= 0:
            return ""
        recent = chat_history[-(max_turns * 2):]
        lines = []
        for msg in recent:
            role_label = "User" if msg.get("role") == "user" else "Assistant"
            content = (msg.get("content") or "").strip()
            if len(content) > 300:
                content = content[:300] + "…"
            if content:
                lines.append(f"{role_label}: {content}")
        return "\n".join(lines)

    @staticmethod
    def _build_sources(docs: List, question: str) -> List[dict]:
        """Build rich source entries (file, page, and a short excerpt with
        query terms bolded) instead of just a bare filename — this is what
        lets someone actually verify an answer against the source text
        without having to go dig up the original document."""
        query_terms = [t_ for t_ in re.findall(r"\w+", question, flags=re.UNICODE) if len(t_) > 2]
        seen = set()
        entries = []
        for doc in docs:
            file_name = doc.metadata.get("file_name", "Unknown")
            page = doc.metadata.get("page", "")
            key = (file_name, page)
            if key in seen:
                continue
            seen.add(key)

            snippet = doc.page_content.strip().replace("\n", " ")
            if len(snippet) > 220:
                snippet = snippet[:220] + "…"
            highlighted = snippet
            for term in sorted(set(query_terms), key=len, reverse=True)[:12]:
                try:
                    highlighted = re.sub(
                        rf"\b({re.escape(term)})\b", r"**\1**", highlighted, flags=re.IGNORECASE | re.UNICODE
                    )
                except re.error:
                    continue

            entries.append({
                "file_name": file_name,
                "page": page,
                "snippet": highlighted,
                "ocr": bool(doc.metadata.get("ocr")),
            })
        return entries

    def query(self, question: str, language: str = "en", file_names: Optional[List[str]] = None):
        if not self.llm_manager.is_ready():
            return None, "LLM not initialized. Please load the LLM model first."

        try:
            docs = self.retriever.retrieve(question, llm=self.llm_manager, file_names=file_names)

            if not docs:
                return {
                    "result": no_answer_message(language),
                    "source_documents": [],
                }, None

            context = "\n\n".join(doc.page_content for doc in docs)
            prompt = self.llm_manager.build_prompt(context, question, language=language)
            answer = self.llm_manager.invoke(prompt)
            if not isinstance(answer, str):
                answer = str(answer)

            return {"result": answer.strip(), "source_documents": docs}, None
        except Exception as e:
            logger.error(f"Error querying: {e}")
            return None, str(e)

    @staticmethod
    def _decorate_sources(sources: List[dict], question: str) -> List[dict]:
        """Add a short highlighted snippet and legacy keys so exports and old renderers still work."""
        terms = sorted({w for w in re.findall(r"\w+", question, flags=re.UNICODE) if len(w) > 2},
                       key=len, reverse=True)[:12]
        for s in sources:
            snippet = s["text"].strip().replace("\n", " ")
            snippet = snippet[:220] + ("…" if len(snippet) > 220 else "")
            for term in terms:
                try:
                    snippet = re.sub(rf"\b({re.escape(term)})\b", r"**\1**", snippet,
                                     flags=re.IGNORECASE | re.UNICODE)
                except re.error:
                    continue
            s["snippet"] = snippet
            s["page"] = s.get("pages") or ""
        return sources

    def _standalone_query(self, question: str, chat_history: Optional[List[dict]]) -> str:
        """Turn a follow-up ('what about its price?') into a self-contained search query."""
        if not self.config.config.get("use_condense", True) or not chat_history:
            return question
        if not (len(question.split()) < 12 or FOLLOWUP_HINT.search(question)):
            return question
        plain = [m for m in chat_history if m.get("role") in ("user", "assistant") and m.get("content")]
        if self.llm_manager.is_ready():
            try:
                transcript = "\n".join(f"{m['role'].upper()}: {m['content'][:600]}" for m in plain[-6:])
                out = self.llm_manager.invoke(
                    f"{CONDENSE_PROMPT}\n\nCONVERSATION:\n{transcript}\n\nLAST MESSAGE: {question}\n\nSTANDALONE QUERY:"
                )
                out = out.splitlines()[0].strip(" \"'") if out else ""
                if 3 <= len(out) <= 400:
                    return out
            except Exception:
                logger.exception("Follow-up condensing failed; using a heuristic instead")
        prev = [m["content"] for m in plain if m["role"] == "user"]
        return f"{prev[-1]} {question}" if prev else question

    def verify_answer(self, answer: str, sources: List[dict]) -> List[dict]:
        """Embed each answer sentence and compare it with the passages' sentences: flags unsupported,
        mis-cited, uncited and invalid citations. A screen for hallucination, not formal entailment."""
        cfg = self.config.config
        if not cfg.get("verify_answers", True) or not sources:
            return []
        import numpy as np
        sents = split_sentences(answer, 25)
        ctx = {s["n"]: s for s in sources}
        psents = [(s["n"], p) for s in sources for p in split_sentences(s["text"], 20)][:400]
        if not sents or not psents:
            return []
        emb = self.vector_manager.embeddings
        a = np.asarray(emb.embed_documents([_CITE_RE.sub("", s).strip() for s in sents]), dtype="float32")
        b = np.asarray(emb.embed_documents([p for _, p in psents]), dtype="float32")
        a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-9)
        b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-9)
        sims = a @ b.T
        ns = np.array([n for n, _ in psents])
        thr = float(cfg.get("verify_threshold", 0.45))
        out = []
        for i, s in enumerate(sents):
            cites = sorted(cited_numbers(s))
            valid = [c for c in cites if c in ctx]
            invalid = [c for c in cites if c not in ctx]
            row = sims[i]
            best_all, best_n = float(row.max()), int(ns[int(row.argmax())])
            mask = np.isin(ns, valid) if valid else None
            cited_best = float(row[mask].max()) if mask is not None and mask.any() else None
            if invalid:
                status = "invalid"
            elif not cites:
                status = "uncited" if best_all >= thr else "unsupported"
            elif cited_best is not None and cited_best >= thr:
                status = "ok"
            else:
                status = "miscited" if best_all >= thr else "unsupported"
            out.append({"claim": s, "status": status, "support": round(best_all, 2), "cites": cites,
                        "best_source": best_n})
        return out

    def query_stream(self, question: str, language: str = "en",
                     chat_history: Optional[List[dict]] = None,
                     file_names: Optional[List[str]] = None):
        """Generator: yields dicts as the answer is produced.

          {"type": "meta", "sources": [...], "retrieval_time": float, "from_cache": bool, "standalone": str}
          {"type": "token", "text": str}                (repeated)
          {"type": "done", "answer": str, "generation_time": float, "total_time": float,
           "backend": "gguf" | "openai" | "cache" | "retrieval" | None}
        or {"type": "error", "message": str}.

        Pipeline: condense follow-ups -> (cache) -> hybrid retrieval (vector + FTS5 + entity graph, RRF)
        -> rerank -> MMR -> neighbour expansion -> numbered evidence -> LLM. With no LLM loaded the most
        relevant passages are returned instead of a written answer. If nothing relevant is found the
        fixed "I don't know" message is returned and the LLM is NOT called."""
        t_start = time.time()
        cfg = self.config.config
        llm_ready = self.llm_manager.is_ready()

        try:
            standalone = self._standalone_query(question, chat_history)
        except Exception:
            logger.exception("Standalone-query step failed")
            standalone = question

        use_cache = cfg.get("use_answer_cache", True) and llm_ready
        cache_key = self._cache_key(standalone, language, file_names)
        fingerprint = self._corpus_fingerprint()

        if use_cache:
            cached = self.persistent_store.get_cached_answer(cache_key, fingerprint)
            if cached is not None:
                yield {"type": "meta", "sources": cached["sources"], "retrieval_time": 0.0,
                       "from_cache": True, "standalone": standalone}
                yield {"type": "token", "text": cached["answer"]}
                yield {"type": "done", "answer": cached["answer"], "generation_time": 0.0,
                       "total_time": time.time() - t_start, "backend": "cache"}
                return

        try:
            docs = self.retriever.retrieve(standalone, llm=self.llm_manager if llm_ready else None,
                                           file_names=file_names)
        except Exception as e:
            logger.error(f"Error retrieving: {e}")
            yield {"type": "error", "message": str(e)}
            return
        retrieval_time = time.time() - t_start

        if not docs:
            no_result_msg = no_answer_message(language)
            yield {"type": "meta", "sources": [], "retrieval_time": retrieval_time,
                   "from_cache": False, "standalone": standalone}
            yield {"type": "token", "text": no_result_msg}
            yield {"type": "done", "answer": no_result_msg, "generation_time": 0.0,
                   "total_time": time.time() - t_start, "backend": None}
            return

        sources = self._decorate_sources(self.retriever.build_context(docs), question)

        if not llm_ready:
            answer = extractive_answer(sources, language)
            yield {"type": "meta", "sources": sources, "retrieval_time": retrieval_time,
                   "from_cache": False, "standalone": standalone}
            yield {"type": "token", "text": answer}
            yield {"type": "done", "answer": answer, "generation_time": 0.0,
                   "total_time": time.time() - t_start, "backend": "retrieval"}
            return

        history_text = ""
        if cfg.get("use_chat_memory", True):
            history_text = self._build_history_text(chat_history, int(cfg.get("memory_turns", 3)))
        prompt = self.llm_manager.build_prompt(build_evidence(sources), question, language=language,
                                               history_text=history_text)

        yield {"type": "meta", "sources": sources, "retrieval_time": retrieval_time,
               "from_cache": False, "standalone": standalone}

        t_gen_start = time.time()
        chunks = []
        try:
            for text in self.llm_manager.stream(prompt):
                chunks.append(text)
                yield {"type": "token", "text": text}
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            yield {"type": "error", "message": str(e)}
            return
        generation_time = time.time() - t_gen_start
        answer = "".join(chunks).strip()

        if use_cache and answer:
            try:
                self.persistent_store.set_cached_answer(cache_key, standalone, answer, sources, fingerprint)
            except Exception:
                logger.exception("Failed to write answer cache")

        yield {"type": "done", "answer": answer, "generation_time": generation_time,
               "total_time": time.time() - t_start, "backend": self.llm_manager.last_backend_used}

    def delete_document_full(self, file_path: Path) -> bool:
        file_name = file_path.name
        disk_ok = self.doc_manager.delete_document(file_path)
        index_ok = self.vector_manager.delete_documents_by_filename(file_name)
        self.persistent_store.remove_by_filename(file_name)
        self.retriever.mark_dirty()
        return disk_ok and (index_ok or self.vector_manager.get_document_count() == 0)

    def clear_knowledge_base(self):
        try:
            self.vector_manager.clear_vectorstore()

            for doc in self.doc_manager.get_all_documents():
                self.doc_manager.delete_document(doc)

            self.persistent_store.clear()
            self.retriever.mark_dirty()
            return True
        except Exception as e:
            logger.error(f"Error clearing knowledge base: {e}")
            return False

    def get_manifest_stats(self) -> dict:
        return self.persistent_store.manifest_stats()
