"""
Step 4 of the pipeline: retrieved chunks + question -> grounded answer.

    .venv\\Scripts\\python.exe answer_generation_pdf.py     # interactive Q&A in the terminal

LLM: llama3 running locally in Ollama. Before starting:
    ollama pull llama3
    ollama serve            # if the Ollama app isn't already running
"""

import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from answer_cache import AnswerCache, same_numbers
from retrieval_pipeline_pdf import PdfRetriever, embed

load_dotenv()

PROVIDER = "ollama"
OLLAMA_MODEL = "llama3"
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
HISTORY_TURNS = 3
MULTI_QUERY_COUNT = 3     # extra search phrasings the LLM writes for each question
DEFAULT_K = 1             # chunks sent to the LLM for the answer
SEMANTIC_THRESHOLD = 0.95  # a saved question must be this similar (cosine) to reuse its answer
SOURCE_MIN_SCORE = 0.0     # saved sources must still rerank above this for the new question

SYSTEM_PROMPT = """You answer questions about the user's PDF documents.
Use ONLY the numbered context passages below. Cite passages inline like [1] or [2][3].
If the context does not contain the answer, say you could not find it in the documents.
Tables are in Markdown; keep numbers exactly as written."""

_cache: Optional[AnswerCache] = None


def get_cache() -> AnswerCache:
    global _cache
    if _cache is None:
        _cache = AnswerCache()
    return _cache


def get_llm(provider: str = PROVIDER):
    from langchain_ollama import ChatOllama
    return ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.1)


def generate_queries(llm, question: str, n: int = MULTI_QUERY_COUNT) -> List[str]:
    """Ask the LLM for other ways to phrase the question, so the search covers more wording."""
    from langchain_core.messages import HumanMessage

    prompt = (
        f"Write {n} different search queries that would find passages answering this question "
        "in a document. Use different wording and synonyms. Output one query per line, "
        "with no numbering and no other text.\n\nQuestion: " + question
    )
    try:
        reply = llm.invoke([HumanMessage(prompt)])
        text = reply.content if isinstance(reply.content, str) else reply.text
    except Exception:
        return [question]    # the answer step reports LLM errors; search with the question alone

    variants = [line.strip(" -*\t0123456789.)") for line in text.splitlines()]
    variants = [v for v in variants if v and not v.endswith(":") and v.lower() != question.lower()]
    return [question] + variants[:n]


def format_context(hits: List[Dict[str, Any]]) -> str:
    blocks = []
    for i, h in enumerate(hits, 1):
        pages = f"p{h['page_start']}" if h["page_start"] == h["page_end"] else f"p{h['page_start']}-{h['page_end']}"
        where = f" | {h['section_path']}" if h.get("section_path") else ""
        blocks.append(f"[{i}] {h['source']} {pages}{where} ({h['chunk_type']})\n{h['content']}")
    return "\n\n---\n\n".join(blocks)


class PdfAnswerer:
    def __init__(self, provider: str = PROVIDER, retriever: Optional[PdfRetriever] = None):
        self.provider = provider
        self.llm = get_llm(provider)
        self.retriever = retriever or PdfRetriever()
        self.history: List[Dict[str, str]] = []

    def reset(self):
        self.history = []

    def _reuse_cached(
        self, question: str, vec: List[float], k: int, chunk_type: str, version: str
    ) -> Optional[Dict[str, Any]]:
        """Return a saved answer if it still fits this question, otherwise None."""
        cache = get_cache()
        hit = cache.find_similar(vec, k, chunk_type, version, SEMANTIC_THRESHOLD)
        if hit is None or not same_numbers(question, hit["question"]):
            return None

        # The saved sources must still be relevant to the new wording
        scores = self.retriever.score_chunks(question, [s["chunk_id"] for s in hit["sources"]])
        top = max(scores, default=None)
        if top is None or top < SOURCE_MIN_SCORE:
            return None

        cache.log(question, True, top)
        self.history.append({"question": question, "answer": hit["answer"]})
        return {
            "question": question, "answer": hit["answer"], "provider": self.provider,
            "queries": hit["queries"], "sources": hit["sources"],
            "from_cache": True, "answer_id": hit["id"], "similarity": round(hit["similarity"], 4),
        }

    def answer(self, question: str, k: int = DEFAULT_K, chunk_type: Optional[str] = None) -> Dict[str, Any]:
        # Short follow-ups ("and in 2023?") depend on the conversation, so they skip the cache
        is_followup = bool(self.history) and len(question.split()) < 8
        version = self.retriever.version
        chunk_type_key = chunk_type or ""
        vec = embed([question])[0]

        if not is_followup:
            cached = self._reuse_cached(question, vec, k, chunk_type_key, version)
            if cached is not None:
                return cached

        # Short follow-ups need the previous question to retrieve well
        query = question
        if is_followup:
            query = f"{self.history[-1]['question']} {question}"

        queries = generate_queries(self.llm, query)
        hits = self.retriever.search_fused(queries, k=k, chunk_type=chunk_type)
        sources = [
            {
                "n": i, "chunk_id": h["chunk_id"], "source": h["source"],
                "page_start": h["page_start"], "page_end": h["page_end"],
                "chunk_type": h["chunk_type"], "score": h["score"], "vector_score": h["vector_score"],
                "image": h.get("image"),
            }
            for i, h in enumerate(hits, 1)
        ]

        if not hits:
            text = "No matching passages were found in the uploaded documents."
        else:
            from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

            messages = [SystemMessage(SYSTEM_PROMPT + "\n\nCONTEXT:\n\n" + format_context(hits))]
            for turn in self.history[-HISTORY_TURNS:]:
                messages += [HumanMessage(turn["question"]), AIMessage(turn["answer"])]
            messages.append(HumanMessage(question))
            try:
                response = self.llm.invoke(messages)
            except Exception as e:
                raise RuntimeError(
                    f"Ollama ({OLLAMA_MODEL}) at {OLLAMA_BASE_URL} failed: {e}. "
                    f"Is Ollama running, and did you run 'ollama pull {OLLAMA_MODEL}'?"
                ) from e
            text = response.content if isinstance(response.content, str) else response.text

        cache = get_cache()
        cache.log(question, False, sources[0]["score"] if sources else None)
        answer_id = None
        if hits and not is_followup:
            answer_id = cache.store(question, vec, k, chunk_type_key, version, text, queries, sources)

        self.history.append({"question": question, "answer": text})
        return {"question": question, "answer": text, "provider": self.provider,
                "queries": queries, "sources": sources, "from_cache": False, "answer_id": answer_id}


if __name__ == "__main__":
    answerer = PdfAnswerer()
    print(f"Ollama model: {OLLAMA_MODEL}. Empty line to quit.")
    while True:
        q = input("\nQ: ").strip()
        if not q:
            break
        result = answerer.answer(q)
        print(f"\nA: {result['answer']}")
        for s in result["sources"]:
            print(f"  [{s['n']}] {s['source']} p{s['page_start']} ({s['chunk_type']}, {s['score']:.2f})")
