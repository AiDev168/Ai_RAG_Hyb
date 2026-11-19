"""Text / query / citation helpers: FTS5 expressions, entity extraction, evidence building."""

import re
from typing import List, Optional

from .constants import RETRIEVAL_ONLY_TITLES, no_answer_message


def _normalize_search_text(text: str) -> str:
    """Normalize text before it goes into (or is queried against) the FTS5
    keyword index. Persian text in particular varies a lot in ways that
    would otherwise break exact keyword matching: the zero-width
    non-joiner used inside compound words, and Arabic-vs-Persian forms of
    the same letters (ي/ی, ك/ک) that different keyboards/fonts produce."""
    if not text:
        return text
    text = text.replace("\u200c", " ")  # ZWNJ -> space
    text = text.replace("\u064a", "\u06cc").replace("\u0643", "\u06a9")  # Arabic ye/kaf -> Persian
    return text


def _fts5_match_expression(query: str) -> Optional[str]:
    """Build a safe FTS5 MATCH expression from free-text user input.
    Tokens are individually double-quoted so punctuation, parentheses, or
    other FTS5 query-syntax characters in the user's question can't raise a
    syntax error — each token is matched as a literal phrase, OR'd together."""
    query = _normalize_search_text(query)
    tokens = re.findall(r"\w+", query, flags=re.UNICODE)
    tokens = [tok for tok in tokens if tok]
    if not tokens:
        return None
    escaped = [tok.replace('"', '""') for tok in tokens[:32]]
    return " OR ".join(f'"{tok}"' for tok in escaped)


# --------------------------------------------------------------------------
# AURA additions — entity graph, query helpers, citation utilities
# --------------------------------------------------------------------------
QUERY_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "is", "are", "was", "were", "be",
    "by", "as", "at", "it", "its", "this", "that", "these", "those", "what", "which", "who", "whom",
    "how", "why", "when", "where", "does", "do", "did", "can", "could", "should", "would", "about",
    "from", "into", "than", "then", "there", "their", "them", "tell", "me", "us", "you", "your",
}

STOP_ENTITIES = {"The", "This", "That", "These", "Those", "However", "Therefore", "Figure", "Table", "Section",
                 "Chapter", "Abstract", "Introduction", "Conclusion", "References", "Keywords", "Copyright",
                 "What", "How", "Why", "When", "Where", "Which", "Who", "Compare", "Explain", "Does", "In",
                 "For", "We", "It", "Our", "A", "An", "As", "On", "If", "Of", "And", "But", "Also", "Thus"}
_W = r"[A-Z][A-Za-z0-9&\-]*(?:\.[A-Za-z0-9&\-]+)*"
ENTITY_RE = re.compile(rf"\b{_W}(?:\s+{_W}){{0,3}}\b")


def extract_entities(text: str, limit: int = 40) -> List[str]:
    """Capitalised-phrase extractor that skips sentence-initial single words (Latin script only;
    Persian text has no capitalisation, so the keyword + vector paths carry it). Swap for spaCy NER if desired."""
    out, seen = [], set()
    for m in ENTITY_RE.finditer(text or ""):
        words = m.group().split()
        stripped = False
        while words and words[0].strip(" .,-&") in STOP_ENTITIES:
            words.pop(0)
            stripped = True
        ent = " ".join(words).strip(" .,-&")
        if len(ent) < 3 or ent in STOP_ENTITIES:
            continue
        if len(ent.split()) == 1 and not (ent.isupper() or any(c.isdigit() for c in ent)) and not stripped:
            j = m.start() - 1
            while j >= 0 and text[j] in " \t":
                j -= 1
            if j < 0 or text[j] in ".!?:;\n\"(":
                continue  # sentence-initial capital, probably just grammar
        key = ent.casefold()
        if key not in seen:
            seen.add(key)
            out.append(ent)
    return out[:limit]


FOLLOWUP_HINT = re.compile(
    r"\b(it|its|they|them|their|this|that|these|those|he|she|his|her|there|above|previous|former|latter|same)\b"
    r"|آن|این|آنها|همان|قبلی|بالا", re.I)

CONDENSE_PROMPT = ("Rewrite the user's LAST message as a single, standalone search query that can be understood "
                   "without the conversation. Resolve pronouns and references using the conversation. Keep names, "
                   "numbers and technical terms. Keep the same language as the message. "
                   "Output ONLY the rewritten query, nothing else.")

HYDE_PROMPT = ("Write a short (2-3 sentence) passage, in the style of a document excerpt, that would directly answer "
               "the user's question. It does not need to be correct. Output only the passage.")

_CITE_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def cited_numbers(text: str) -> set:
    return {int(n) for grp in _CITE_RE.findall(text or "") for n in re.split(r"\s*,\s*", grp)}


def split_sentences(text: str, min_len: int = 25) -> List[str]:
    parts = re.split(r"(?<=[.!?؟])\s+|\n+", text or "")
    return [re.sub(r"^[\-\*•\d.\)\s]+", "", s).strip() for s in parts if len(s.strip()) >= min_len]


def page_label_from(pages: List[int]) -> Optional[str]:
    """pages are 0-based loader page indexes; shown 1-based."""
    if not pages:
        return None
    lo, hi = min(pages) + 1, max(pages) + 1
    return f"p. {lo}" if lo == hi else f"pp. {lo}–{hi}"


def _join_overlapping(texts: List[str]) -> str:
    """Concatenate consecutive chunks, trimming the splitter's overlap so text isn't repeated."""
    out = texts[0] if texts else ""
    for nxt in texts[1:]:
        best = 0
        for k in range(min(len(out), len(nxt), 400), 24, -1):
            if out.endswith(nxt[:k]):
                best = k
                break
        out = out + (nxt[best:] if best else "\n" + nxt)
    return out


def build_evidence(sources: List[dict]) -> str:
    blocks = []
    for s in sources:
        where = s["file_name"] + (f", {s['pages']}" if s.get("pages") else "")
        blocks.append(f"[{s['n']}] {where}\n{s['text'].strip()}")
    return "\n\n".join(blocks)


def extractive_answer(sources: List[dict], language: str = "en") -> str:
    if not sources:
        return no_answer_message(language)
    title = RETRIEVAL_ONLY_TITLES.get(language, RETRIEVAL_ONLY_TITLES["en"])
    lines = [f"**{title}**\n"]
    for s in sources[:4]:
        snippet = re.sub(r"\s+", " ", s["text"]).strip()[:420]
        lines.append(f"- {snippet}… [{s['n']}]")
    return "\n".join(lines)
