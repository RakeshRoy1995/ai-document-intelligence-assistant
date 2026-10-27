"""
Answer cache: a repeated question gets its saved answer instead of a new search and LLM call.

Stored in data/answer_cache.sqlite3 (one file, no server needed). Safeguards that keep
reused answers accurate (see PdfAnswerer._reuse_cached in answer_generation_pdf.py):

    1. Semantic match: the new question must be very close to a saved question.
    2. Same numbers: "Q3" and "Q4" look alike to an embedding model, so the numbers must match.
    3. Still supported: the saved sources must still rerank well for the new question.

Feedback (thumbs up / down) adjusts entries: enough thumbs-up makes an answer verified and
keeps it; enough thumbs-down takes it out of use. Each entry belongs to a document version,
so uploading or removing a PDF makes old answers stop matching.
"""

import json
import os
import re
import sqlite3
import threading
import time
from contextlib import closing
from typing import Any, Dict, List, Optional

DATA_DIR = os.environ.get("DATA_DIR", "data")
CACHE_PATH = os.path.join(DATA_DIR, "answer_cache.sqlite3")
CACHE_TTL_DAYS = 7      # unverified answers expire after this many days
VERIFY_VOTES = 2        # this many thumbs-up makes an answer verified (it never expires)
DISABLE_VOTES = 2       # this many thumbs-down, and more down than up, takes it out of use

SCHEMA = """
CREATE TABLE IF NOT EXISTS answers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question TEXT NOT NULL,
    embedding TEXT NOT NULL,
    k INTEGER NOT NULL,
    chunk_type TEXT NOT NULL,
    version TEXT NOT NULL,
    answer TEXT NOT NULL,
    queries TEXT NOT NULL,
    sources TEXT NOT NULL,
    upvotes INTEGER NOT NULL DEFAULT 0,
    downvotes INTEGER NOT NULL DEFAULT 0,
    verified INTEGER NOT NULL DEFAULT 0,
    disabled INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_answers_lookup ON answers (k, chunk_type, version, disabled);
CREATE TABLE IF NOT EXISTS query_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question TEXT NOT NULL,
    cache_hit INTEGER NOT NULL,
    top_score REAL,
    created_at REAL NOT NULL
);
"""


def same_numbers(a: str, b: str) -> bool:
    """True when both questions mention exactly the same numbers ("Q3" vs "Q4" -> False)."""
    def numbers(s: str) -> List[str]:
        return sorted(set(re.findall(r"\d+(?:[.,]\d+)?", s)))
    return numbers(a) == numbers(b)


def _decode(row: sqlite3.Row) -> Dict[str, Any]:
    return {
        "id": row["id"],
        "question": row["question"],
        "answer": row["answer"],
        "queries": json.loads(row["queries"]),
        "sources": json.loads(row["sources"]),
        "verified": bool(row["verified"]),
    }


class AnswerCache:
    def __init__(self, path: str = CACHE_PATH):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.path = path
        self.lock = threading.Lock()
        with self.lock, closing(sqlite3.connect(path)) as conn, conn:
            conn.executescript(SCHEMA)

    def _run(self, sql: str, params: tuple = ()) -> List[sqlite3.Row]:
        with self.lock, closing(sqlite3.connect(self.path)) as conn, conn:
            conn.row_factory = sqlite3.Row
            return conn.execute(sql, params).fetchall()

    def _insert(self, sql: str, params: tuple) -> int:
        with self.lock, closing(sqlite3.connect(self.path)) as conn, conn:
            return conn.execute(sql, params).lastrowid

    def find_similar(
        self, vec: List[float], k: int, chunk_type: str, version: str, threshold: float
    ) -> Optional[Dict[str, Any]]:
        """Best saved answer whose question is at least `threshold` similar (cosine) to `vec`."""
        cutoff = time.time() - CACHE_TTL_DAYS * 86400
        rows = self._run(
            "SELECT * FROM answers WHERE k = ? AND chunk_type = ? AND version = ? AND disabled = 0 "
            "AND (verified = 1 OR created_at >= ?)",
            (k, chunk_type, version, cutoff),
        )
        best, best_sim = None, threshold
        for row in rows:
            # embeddings are normalized, so the dot product is the cosine similarity
            sim = sum(a * b for a, b in zip(vec, json.loads(row["embedding"])))
            if sim >= best_sim:
                best, best_sim = row, sim
        if best is None:
            return None
        return {**_decode(best), "similarity": best_sim}

    def store(
        self, question: str, vec: List[float], k: int, chunk_type: str, version: str,
        answer: str, queries: List[str], sources: List[Dict[str, Any]],
    ) -> int:
        return self._insert(
            "INSERT INTO answers (question, embedding, k, chunk_type, version, answer, queries, sources, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (question, json.dumps(vec), k, chunk_type, version, answer,
             json.dumps(queries), json.dumps(sources), time.time()),
        )

    def vote(self, answer_id: int, helpful: bool) -> Optional[Dict[str, Any]]:
        with self.lock, closing(sqlite3.connect(self.path)) as conn, conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM answers WHERE id = ?", (answer_id,)).fetchone()
            if row is None:
                return None
            up = row["upvotes"] + (1 if helpful else 0)
            down = row["downvotes"] + (0 if helpful else 1)
            verified = bool(row["verified"]) or up >= VERIFY_VOTES
            disabled = not verified and down >= DISABLE_VOTES and down > up
            conn.execute(
                "UPDATE answers SET upvotes = ?, downvotes = ?, verified = ?, disabled = ? WHERE id = ?",
                (up, down, int(verified), int(disabled), answer_id),
            )
        return {"answer_id": answer_id, "upvotes": up, "downvotes": down,
                "verified": verified, "disabled": disabled}

    def log(self, question: str, hit: bool, top_score: Optional[float]):
        self._insert(
            "INSERT INTO query_log (question, cache_hit, top_score, created_at) VALUES (?, ?, ?, ?)",
            (question, int(hit), top_score, time.time()),
        )
