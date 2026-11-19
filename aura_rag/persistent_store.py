"""SQLite-backed persistent store: chunks, FTS5 index, manifest, conversations, answer cache."""

import logging
import json
import math
import os
import re
import sqlite3
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from .config import Config
from .dependencies import Document
from .text_utils import (
    QUERY_STOPWORDS,
    _fts5_match_expression,
    _normalize_search_text,
    extract_entities,
)

logger = logging.getLogger(__name__)


class PersistentStore:
    """Disk-backed, incrementally-updatable storage for two things that used
    to live only in memory or in an ever-growing flat file:

      1. The raw chunk corpus + a full-text (FTS5) keyword index, used for
         hybrid search. SQLite FTS5 is C-implemented and disk-backed, so
         keyword search scales to a large corpus without ever loading every
         chunk into Python memory — unlike the previous JSONL + in-memory
         rank_bm25 approach, which had to read and re-tokenize the entire
         corpus on every single change.
      2. An ingestion manifest recording which source files have already
         been indexed (path, size, mtime, status, chunk count). This is
         what makes bulk folder ingestion resumable: if a 50GB import is
         interrupted, restarting it skips every file already marked done
         instead of re-processing the whole corpus.

    One SQLite connection (WAL mode, so reads from the UI don't block an
    ingestion write in progress) backs both concerns.
    """

    def __init__(self, config: Config):
        self.config = config
        self.path = config.db_dir / "store.sqlite3"
        self._local = threading.local()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        # A connection per thread: the background ingestion thread and the
        # Streamlit UI thread must not share a single sqlite3 connection.
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(str(self.path), timeout=30, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            self._local.conn = conn
        return conn

    def _init_schema(self) -> None:
        conn = self._connect()
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_name TEXT NOT NULL,
                source_file TEXT,
                metadata_json TEXT,
                page_content TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_file_name ON chunks(file_name);

            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                page_content,
                content='chunks',
                content_rowid='id',
                tokenize='unicode61 remove_diacritics 2'
            );

            CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
                INSERT INTO chunks_fts(rowid, page_content) VALUES (new.id, new.page_content);
            END;
            CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, page_content) VALUES('delete', old.id, old.page_content);
            END;

            CREATE TABLE IF NOT EXISTS indexed_files (
                path TEXT PRIMARY KEY,
                file_name TEXT,
                size INTEGER,
                mtime REAL,
                status TEXT,
                chunk_count INTEGER DEFAULT 0,
                error TEXT,
                indexed_at TEXT
            );

            CREATE TABLE IF NOT EXISTS answer_cache (
                cache_key TEXT PRIMARY KEY,
                question TEXT,
                answer TEXT,
                sources_json TEXT,
                corpus_fingerprint TEXT,
                created_at TEXT
            );

            CREATE TABLE IF NOT EXISTS entities (
                entity TEXT NOT NULL,
                chunk_id INTEGER NOT NULL,
                file_name TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(entity);
            CREATE INDEX IF NOT EXISTS idx_entities_file ON entities(file_name);

            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created REAL NOT NULL,
                updated REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conv_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                payload TEXT NOT NULL DEFAULT '{}',
                ts REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conv_id, id);
            """
        )
        conn.commit()

    # -- answer cache --------------------------------------------------------

    def get_cached_answer(self, cache_key: str, corpus_fingerprint: str) -> Optional[dict]:
        """Return the cached {answer, sources} for this exact question if
        present *and* the corpus hasn't changed since it was cached — a
        cheap chunk-count-based fingerprint is enough to avoid ever serving
        a stale answer after documents are added, removed, or the knowledge
        base is cleared, without needing anything more elaborate."""
        conn = self._connect()
        row = conn.execute(
            "SELECT answer, sources_json, corpus_fingerprint FROM answer_cache WHERE cache_key = ?",
            (cache_key,),
        ).fetchone()
        if row is None:
            return None
        answer, sources_json, cached_fingerprint = row
        if cached_fingerprint != corpus_fingerprint:
            return None
        try:
            sources = json.loads(sources_json) if sources_json else []
        except Exception:
            sources = []
        return {"answer": answer, "sources": sources}

    def set_cached_answer(self, cache_key: str, question: str, answer: str, sources: list, corpus_fingerprint: str) -> None:
        conn = self._connect()
        conn.execute(
            """
            INSERT INTO answer_cache (cache_key, question, answer, sources_json, corpus_fingerprint, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(cache_key) DO UPDATE SET
                answer=excluded.answer, sources_json=excluded.sources_json,
                corpus_fingerprint=excluded.corpus_fingerprint, created_at=excluded.created_at
            """,
            (cache_key, question, answer, json.dumps(sources, ensure_ascii=False), corpus_fingerprint, datetime.now().isoformat()),
        )
        conn.commit()

    def clear_answer_cache(self) -> None:
        conn = self._connect()
        conn.execute("DELETE FROM answer_cache")
        conn.commit()

    def answer_cache_count(self) -> int:
        conn = self._connect()
        return conn.execute("SELECT COUNT(*) FROM answer_cache").fetchone()[0]

    # -- chunk storage / keyword search -----------------------------------

    def append(self, documents) -> None:
        conn = self._connect()
        rows = [
            (
                doc.metadata.get("file_name", ""),
                doc.metadata.get("source_file", ""),
                json.dumps(doc.metadata, ensure_ascii=False),
                _normalize_search_text(doc.page_content),
            )
            for doc in documents
        ]
        if conn.in_transaction:
            conn.commit()
        # One write transaction: the new chunk ids are contiguous, so the entity
        # rows can be attached to them without a second pass over the corpus.
        conn.execute("BEGIN IMMEDIATE")
        try:
            before = conn.execute("SELECT COALESCE(MAX(id), 0) FROM chunks").fetchone()[0]
            conn.executemany(
                "INSERT INTO chunks (file_name, source_file, metadata_json, page_content) VALUES (?, ?, ?, ?)",
                rows,
            )
            new_rows = conn.execute(
                "SELECT id, file_name, page_content FROM chunks WHERE id > ?", (before,)
            ).fetchall()
            ent_rows = [(e.casefold(), cid, fn) for cid, fn, content in new_rows for e in extract_entities(content)]
            if ent_rows:
                conn.executemany("INSERT INTO entities (entity, chunk_id, file_name) VALUES (?, ?, ?)", ent_rows)
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def remove_by_filename(self, file_name: str) -> None:
        conn = self._connect()
        conn.execute("DELETE FROM entities WHERE file_name = ?", (file_name,))
        conn.execute("DELETE FROM chunks WHERE file_name = ?", (file_name,))
        conn.commit()

    def clear(self) -> None:
        conn = self._connect()
        conn.executescript(
            "DELETE FROM chunks; DELETE FROM chunks_fts; DELETE FROM indexed_files; DELETE FROM entities;"
        )
        conn.commit()

    def count(self) -> int:
        conn = self._connect()
        return conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    def list_file_names(self) -> List[str]:
        """Distinct file names that currently have indexed chunks (covers both
        uploaded files and bulk-imported folder files)."""
        conn = self._connect()
        rows = conn.execute(
            "SELECT DISTINCT file_name FROM chunks WHERE file_name != '' ORDER BY file_name COLLATE NOCASE"
        ).fetchall()
        return [r[0] for r in rows]

    def search_keyword(self, query: str, k: int = 10, file_names: Optional[List[str]] = None) -> List:
        """FTS5 keyword search, ranked by SQLite's built-in bm25() function.
        Runs entirely on disk/index — never loads the corpus into memory.
        If file_names is given, only chunks from those files are searched."""
        if Document is None:
            return []
        match_expr = _fts5_match_expression(query)
        if not match_expr:
            return []
        conn = self._connect()
        sql = """
            SELECT c.page_content, c.metadata_json, bm25(chunks_fts) AS score
            FROM chunks_fts
            JOIN chunks c ON c.id = chunks_fts.rowid
            WHERE chunks_fts MATCH ?
        """
        params: list = [match_expr]
        if file_names:
            sql += " AND c.file_name IN (%s)" % ",".join("?" * len(file_names))
            params.extend(file_names)
        sql += " ORDER BY score LIMIT ?"
        params.append(k)
        try:
            rows = conn.execute(sql, params).fetchall()
        except sqlite3.OperationalError:
            logger.exception("FTS5 keyword search failed for query")
            return []
        results = []
        for page_content, metadata_json, _score in rows:
            try:
                metadata = json.loads(metadata_json) if metadata_json else {}
            except Exception:
                metadata = {}
            results.append(Document(page_content=page_content, metadata=metadata))
        return results

    # -- entity graph ---------------------------------------------------------

    def entity_count(self) -> int:
        return self._connect().execute("SELECT COUNT(*) FROM entities").fetchone()[0]

    def rebuild_entities(self) -> int:
        """Re-extract entities for every stored chunk (needed once for documents
        indexed before the entity graph existed)."""
        conn = self._connect()
        if conn.in_transaction:
            conn.commit()
        conn.execute("DELETE FROM entities")
        rows = conn.execute("SELECT id, file_name, page_content FROM chunks").fetchall()
        batch: list = []
        for cid, fn, content in rows:
            batch.extend((e.casefold(), cid, fn) for e in extract_entities(content))
            if len(batch) >= 5000:
                conn.executemany("INSERT INTO entities (entity, chunk_id, file_name) VALUES (?, ?, ?)", batch)
                batch = []
        if batch:
            conn.executemany("INSERT INTO entities (entity, chunk_id, file_name) VALUES (?, ?, ?)", batch)
        conn.commit()
        return len(rows)

    def _docs_for_ids(self, ids: List[int]) -> List:
        if not ids or Document is None:
            return []
        conn = self._connect()
        found: dict = {}
        for i in range(0, len(ids), 500):
            part = ids[i:i + 500]
            q = f"SELECT id, page_content, metadata_json FROM chunks WHERE id IN ({','.join('?' * len(part))})"
            for cid, content, mj in conn.execute(q, part):
                try:
                    meta = json.loads(mj) if mj else {}
                except Exception:
                    meta = {}
                found[cid] = Document(page_content=content, metadata=meta)
        return [found[i] for i in ids if i in found]

    def search_entities(self, query: str, k: int = 10, file_names: Optional[List[str]] = None) -> List:
        """Chunks sharing named entities with the question, weighted by entity rarity (IDF)."""
        terms = {e.casefold() for e in extract_entities(query)}
        if not terms:
            terms = {w for w in re.findall(r"[\w\-]{4,}", _normalize_search_text(query).casefold())
                     if w not in QUERY_STOPWORDS}
        if not terms:
            return []
        conn = self._connect()
        total = max(1, self.count())
        scores: dict = {}
        for term in terms:
            sql = "SELECT chunk_id, COUNT(*) FROM entities WHERE entity = ?"
            params: list = [term]
            if file_names:
                sql += " AND file_name IN (%s)" % ",".join("?" * len(file_names))
                params.extend(file_names)
            sql += " GROUP BY chunk_id"
            rows = conn.execute(sql, params).fetchall()
            if not rows:
                continue
            idf = math.log(1 + total / len(rows))
            for cid, n in rows:
                scores[cid] = scores.get(cid, 0.0) + idf * (1 + 0.2 * min(n, 5))
        top = [cid for cid, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]]
        return self._docs_for_ids(top)

    # -- neighbour lookup (chunk ids are sequential within a file) -------------

    def find_chunk_id(self, file_name: str, content: str) -> Optional[int]:
        row = self._connect().execute(
            "SELECT id FROM chunks WHERE file_name = ? AND page_content = ? LIMIT 1",
            (file_name, _normalize_search_text(content)),
        ).fetchone()
        return row[0] if row else None

    def fetch_range(self, file_name: str, lo: int, hi: int) -> List[tuple]:
        rows = self._connect().execute(
            "SELECT id, page_content, metadata_json FROM chunks WHERE file_name = ? AND id BETWEEN ? AND ? ORDER BY id",
            (file_name, lo, hi),
        ).fetchall()
        out = []
        for cid, content, mj in rows:
            try:
                meta = json.loads(mj) if mj else {}
            except Exception:
                meta = {}
            out.append((cid, content, meta))
        return out

    # -- conversations ---------------------------------------------------------

    def new_conversation(self, title: str = "New chat") -> str:
        import hashlib
        conn = self._connect()
        cid = hashlib.sha1(os.urandom(16)).hexdigest()[:12]
        now = time.time()
        conn.execute("INSERT INTO conversations(id, title, created, updated) VALUES (?,?,?,?)", (cid, title, now, now))
        conn.commit()
        return cid

    def list_conversations(self) -> List[dict]:
        rows = self._connect().execute(
            """SELECT c.id, c.title, c.updated,
                      (SELECT COUNT(*) FROM messages m WHERE m.conv_id = c.id) AS n
               FROM conversations c ORDER BY c.updated DESC LIMIT 100"""
        ).fetchall()
        return [{"id": r[0], "title": r[1], "updated": r[2], "n": r[3]} for r in rows]

    def add_message(self, conv_id: str, role: str, content: str, payload: Optional[dict] = None) -> None:
        conn = self._connect()
        conn.execute("INSERT INTO messages(conv_id, role, content, payload, ts) VALUES (?,?,?,?,?)",
                     (conv_id, role, content, json.dumps(payload or {}, default=str, ensure_ascii=False), time.time()))
        conn.execute("UPDATE conversations SET updated=? WHERE id=?", (time.time(), conv_id))
        conn.commit()

    def load_messages(self, conv_id: str) -> List[dict]:
        out = []
        for role, content, payload in self._connect().execute(
                "SELECT role, content, payload FROM messages WHERE conv_id=? ORDER BY id", (conv_id,)):
            try:
                extra = json.loads(payload or "{}")
            except Exception:
                extra = {}
            out.append({"role": role, "content": content, **extra})
        return out

    def rename_conversation(self, conv_id: str, title: str) -> None:
        conn = self._connect()
        conn.execute("UPDATE conversations SET title=? WHERE id=?", (title.strip() or "Untitled", conv_id))
        conn.commit()

    def clear_conversation(self, conv_id: str) -> None:
        conn = self._connect()
        conn.execute("DELETE FROM messages WHERE conv_id=?", (conv_id,))
        conn.commit()

    def delete_conversation(self, conv_id: str) -> None:
        conn = self._connect()
        conn.execute("DELETE FROM messages WHERE conv_id=?", (conv_id,))
        conn.execute("DELETE FROM conversations WHERE id=?", (conv_id,))
        conn.commit()

    # -- ingestion manifest (resumable bulk import) ------------------------

    def is_up_to_date(self, path: Path) -> bool:
        """True if this exact file (by path, size and mtime) was already
        successfully indexed — lets a bulk import skip unchanged files on
        resume instead of re-processing a 50GB corpus from scratch."""
        conn = self._connect()
        row = conn.execute(
            "SELECT size, mtime, status FROM indexed_files WHERE path = ?",
            (str(path),),
        ).fetchone()
        if row is None:
            return False
        size, mtime, status = row
        try:
            st_ = path.stat()
        except OSError:
            return False
        return status == "done" and size == st_.st_size and abs(mtime - st_.st_mtime) < 1.0

    def mark_file(self, path: Path, status: str, chunk_count: int = 0, error: str = "") -> None:
        conn = self._connect()
        try:
            size = path.stat().st_size
            mtime = path.stat().st_mtime
        except OSError:
            size, mtime = 0, 0.0
        conn.execute(
            """
            INSERT INTO indexed_files (path, file_name, size, mtime, status, chunk_count, error, indexed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(path) DO UPDATE SET
                file_name=excluded.file_name, size=excluded.size, mtime=excluded.mtime,
                status=excluded.status, chunk_count=excluded.chunk_count,
                error=excluded.error, indexed_at=excluded.indexed_at
            """,
            (str(path), path.name, size, mtime, status, chunk_count, error, datetime.now().isoformat()),
        )
        conn.commit()

    def manifest_stats(self) -> dict:
        conn = self._connect()
        row = conn.execute(
            """
            SELECT
                COUNT(*) FILTER (WHERE status='done') AS done,
                COUNT(*) FILTER (WHERE status='failed') AS failed,
                COALESCE(SUM(chunk_count) FILTER (WHERE status='done'), 0) AS chunks,
                COALESCE(SUM(size) FILTER (WHERE status='done'), 0) AS bytes
            FROM indexed_files
            """
        ).fetchone()
        return {"done": row[0] or 0, "failed": row[1] or 0, "chunks": row[2] or 0, "bytes": row[3] or 0}

    def failed_files(self, limit: int = 50) -> List[dict]:
        conn = self._connect()
        rows = conn.execute(
            "SELECT path, error, indexed_at FROM indexed_files WHERE status='failed' ORDER BY indexed_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [{"path": r[0], "error": r[1], "indexed_at": r[2]} for r in rows]

    # Retained for compatibility with anything that still wants the full
    # corpus as Document objects (small corpora only — this does load
    # everything into memory, so bulk-imported 50GB corpora should use
    # search_keyword() instead, never this).
    def load_all(self) -> List:
        if Document is None:
            return []
        conn = self._connect()
        rows = conn.execute("SELECT page_content, metadata_json FROM chunks").fetchall()
        docs = []
        for page_content, metadata_json in rows:
            try:
                metadata = json.loads(metadata_json) if metadata_json else {}
            except Exception:
                metadata = {}
            docs.append(Document(page_content=page_content, metadata=metadata))
        return docs


# Backward-compatible alias — earlier revisions of this file called this
# class ChunkStore; keep the name usable in case anything external imports it.
ChunkStore = PersistentStore
