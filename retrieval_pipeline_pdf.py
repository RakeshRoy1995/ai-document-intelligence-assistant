"""
Step 3 of the pipeline: chunks JSON -> Chroma vector DB -> semantic search.

    .venv\\Scripts\\python.exe retrieval_pipeline_pdf.py "what is the main result?"

The Chroma collection is persisted in data/chroma/ and kept in sync with
data/chunks.json: creating a PdfRetriever embeds chunks that are new and
deletes vectors whose chunk no longer exists. Chunk ids are content hashes,
so re-uploading an unchanged PDF re-embeds nothing.

Search is two-stage: Chroma returns CANDIDATES nearest chunks by embedding,
then a cross-encoder reranker scores each (query, chunk) pair and keeps the top k.
"""

import hashlib
import json
import os
import sys
import threading
from typing import Any, Dict, List, Optional

import chromadb

from dotenv import load_dotenv

load_dotenv()

DATA_DIR = os.environ.get("DATA_DIR", "data")
CHUNKS_JSON = os.path.join(DATA_DIR, "chunks.json")
CHROMA_DIR = os.path.join(DATA_DIR, "chroma")
COLLECTION = os.environ.get("CHROMA_COLLECTION", "pdf_chunks")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CANDIDATES = 20                 # chunks fetched from Chroma per query before reranking
RRF_K = 60                      # reciprocal rank fusion constant (standard value)
BATCH_SIZE = 64


# ============================================================
# MODELS (loaded once per process)
# ============================================================

_model = None
_reranker = None
_model_lock = threading.Lock()


def get_embedder():
    global _model
    with _model_lock:
        if _model is None:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
        return _model


def get_reranker():
    global _reranker
    with _model_lock:
        if _reranker is None:
            from sentence_transformers import CrossEncoder
            _reranker = CrossEncoder(RERANKER_MODEL, device="cpu")
        return _reranker


def embed(texts: List[str]) -> List[List[float]]:
    vectors = get_embedder().encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
    return vectors.tolist()


def embedding_text(chunk: Dict[str, Any]) -> str:
    """What gets embedded: section path gives context to short chunks."""
    head = f"{chunk['source']} | {chunk['section_path']}" if chunk.get("section_path") else chunk["source"]
    return f"{head}\n\n{chunk['content']}"


def metadata(chunk: Dict[str, Any]) -> Dict[str, Any]:
    """Chroma metadata must be flat str/int/float/bool, no None."""
    meta = {
        "source": chunk["source"],
        "chunk_type": chunk["chunk_type"],
        "section_path": chunk.get("section_path") or "",
        "page_start": int(chunk.get("page_start") or 0),
        "page_end": int(chunk.get("page_end") or 0),
    }
    if chunk.get("image"):
        meta["image"] = chunk["image"]
    return meta


# ============================================================
# RETRIEVER
# ============================================================

class PdfRetriever:
    def __init__(self, chunks_json: str = CHUNKS_JSON):
        with open(chunks_json, "r", encoding="utf-8") as f:
            self.chunks: List[Dict[str, Any]] = json.load(f)
        self.by_id = {c["chunk_id"]: c for c in self.chunks}
        # Identifies this exact set of chunks; cached answers are only reused for the same version
        self.version = hashlib.sha1("|".join(sorted(self.by_id)).encode("utf-8")).hexdigest()[:16]

        client = chromadb.PersistentClient(path=CHROMA_DIR)
        self.collection = client.get_or_create_collection(
            COLLECTION, metadata={"hnsw:space": "cosine"}
        )
        self.sync()

    def sync(self):
        existing = set(self.collection.get(include=[])["ids"])
        wanted = set(self.by_id)

        stale = list(existing - wanted)
        for i in range(0, len(stale), 5000):
            self.collection.delete(ids=stale[i:i + 5000])

        new = [self.by_id[cid] for cid in self.by_id if cid not in existing]
        for i in range(0, len(new), BATCH_SIZE):
            batch = new[i:i + BATCH_SIZE]
            self.collection.add(
                ids=[c["chunk_id"] for c in batch],
                embeddings=embed([embedding_text(c) for c in batch]),
                documents=[c["content"] for c in batch],
                metadatas=[metadata(c) for c in batch],
            )
        if stale or new:
            print(f"[chroma] +{len(new)} added, -{len(stale)} removed, {self.collection.count()} total")

    def score_chunks(self, query: str, chunk_ids: List[str]) -> List[float]:
        """Rerank specific chunks against a query (used to check a saved answer still fits)."""
        chunks = [self.by_id[cid] for cid in chunk_ids if cid in self.by_id]
        if not chunks:
            return []
        scores = get_reranker().predict(
            [(query, embedding_text(c)) for c in chunks], show_progress_bar=False
        )
        return [float(s) for s in scores]

    def search(
        self,
        query: str,
        k: int = 1,
        chunk_type: Optional[str] = None,
        source: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        print("🚀 ~ PdfRetriever ~ search ~ query:", query)
        return self.search_fused([query], k=k, chunk_type=chunk_type, source=source)

    def search_fused(
        self,
        queries: List[str],
        k: int = 1,
        chunk_type: Optional[str] = None,
        source: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Multi-query search: run every query, fuse the rankings with RRF, then rerank.

        RRF gives a chunk sum(1 / (RRF_K + rank)) over the queries that found it,
        so a chunk that several phrasings agree on rises to the top.
        The reranker then scores the fused pool against queries[0] (the question).
        """
        if not self.chunks or not queries:
            return []

        filters = []
        if chunk_type:
            filters.append({"chunk_type": chunk_type})
        if source:
            filters.append({"source": source})
        where = None if not filters else filters[0] if len(filters) == 1 else {"$and": filters}

        rrf: Dict[str, float] = {}
        best_vector: Dict[str, float] = {}
        for query in queries:
            result = self.collection.query(
                query_embeddings=embed([query]),
                n_results=min(CANDIDATES, len(self.chunks)),
                where=where,
                include=["distances"],
            )
            for rank, (cid, dist) in enumerate(zip(result["ids"][0], result["distances"][0]), 1):
                if cid not in self.by_id:
                    continue
                rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (RRF_K + rank)
                best_vector[cid] = max(best_vector.get(cid, -1.0), 1.0 - float(dist))   # cosine similarity

        pool = sorted(rrf, key=rrf.get, reverse=True)[:CANDIDATES]
        hits = [{**self.by_id[cid], "vector_score": round(best_vector[cid], 4)} for cid in pool]
        if not hits:
            return []

        # Rerank: the cross-encoder reads the question and each chunk together
        scores = get_reranker().predict(
            [(queries[0], embedding_text(h)) for h in hits], show_progress_bar=False
        )
        for h, s in zip(hits, scores):
            h["score"] = round(float(s), 4)
        hits.sort(key=lambda h: h["score"], reverse=True)
        return hits[:k]


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "What is this document about?"
    retriever = PdfRetriever()
    for hit in retriever.search(q, k=5):
        print(f"\n[rerank {hit['score']:.3f} | vector {hit['vector_score']:.3f}] {hit['source']} p{hit['page_start']} ({hit['chunk_type']}) {hit['section_path']}")
        print(hit["content"][:300])
