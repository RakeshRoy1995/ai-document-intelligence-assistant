"""
Web app: upload a PDF, see what was extracted, ask questions about it.

    .venv\\Scripts\\python.exe app.py        ->  http://localhost:8000

Pipeline per upload (runs in a background thread, ~1-3 min per PDF on CPU):
    Docling extraction -> section-aware chunks -> Chroma sync -> ready to ask
"""

import base64
import json
import os
import secrets
import threading
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import chunking_pipeline_pdf as chunking
import ingestion_pipeline_docling as ingestion
from answer_generation_pdf import PROVIDER, PdfAnswerer, get_cache, get_llm
from retrieval_pipeline_pdf import CHUNKS_JSON, PdfRetriever


# ============================================================
# CONFIG
# ============================================================

# Hugging Face Spaces / Docker set these; locally the defaults apply
HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "8000"))

# If set, the browser asks for this password (any username).
# Set it on public deployments so strangers can't use your machine's Ollama.
APP_PASSWORD = os.environ.get("APP_PASSWORD", "")

DOCS_DIR = Path(ingestion.DOCS_PATH)
IMAGE_DIR = Path(ingestion.IMAGE_DIR)
LAYOUT_JSON = Path(ingestion.OUTPUT_JSON)
STATIC_DIR = Path("static")

MAX_UPLOAD_MB = 50


# ============================================================
# STATE
# ============================================================

class State:
    """Shared by all requests. One pipeline job runs at a time."""

    def __init__(self):
        self.lock = threading.Lock()          # guards JSON files + retriever swap
        self.job_lock = threading.Lock()      # one Docling job at a time
        self.converter = None                 # Docling models load once, lazily
        self.llm = None
        self.retriever: Optional[PdfRetriever] = None
        self.sessions: Dict[str, PdfAnswerer] = {}
        self.jobs: Dict[str, Dict[str, Any]] = {}

    def get_retriever(self) -> PdfRetriever:
        with self.lock:
            if self.retriever is None:
                if not Path(CHUNKS_JSON).exists():
                    Path(CHUNKS_JSON).write_text("[]", encoding="utf-8")
                self.retriever = PdfRetriever()
            return self.retriever

    def get_session(self, session_id: str) -> PdfAnswerer:
        if self.llm is None:
            self.llm = get_llm(PROVIDER)
        answerer = self.sessions.get(session_id)
        if answerer is None:
            answerer = PdfAnswerer.__new__(PdfAnswerer)   # reuse shared llm + retriever
            answerer.provider = PROVIDER
            answerer.llm = self.llm
            answerer.history = []
            self.sessions[session_id] = answerer
        answerer.retriever = self.get_retriever()        # picks up newly indexed PDFs
        return answerer


state = State()


# ============================================================
# JSON HELPERS
# ============================================================

def read_list(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_list(path: Path, data: List[Dict[str, Any]]):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def replace_source(path: Path, source: str, new_items: List[Dict[str, Any]]):
    """Swap one PDF's records in a merged JSON file, keep the others."""
    kept = [x for x in read_list(path) if x.get("source") != source]
    write_list(path, kept + new_items)


# ============================================================
# PIPELINE JOB
# ============================================================

def run_pipeline(job_id: str, pdf_path: Path):
    job = state.jobs[job_id]
    source = pdf_path.name

    def stage(name: str, progress: int):
        job.update(stage=name, progress=progress)

    try:
        with state.job_lock:
            stage("Loading extraction models", 5)
            if state.converter is None:
                state.converter = ingestion.build_converter()

            stage("Extracting layout, tables, figures, equations", 15)
            result = state.converter.convert(pdf_path)
            items = ingestion.convert_document(result.document, source, IMAGE_DIR)

            stage("Building sections and chunks", 75)
            sections = chunking.build_sections(items)
            chunks = chunking.create_chunks(sections)

            stage("Embedding and indexing", 85)
            with state.lock:
                replace_source(LAYOUT_JSON, source, items)
                all_chunks = [c for c in read_list(Path(CHUNKS_JSON)) if c["source"] != source]
                # chunk_index must stay unique across documents
                for i, c in enumerate(all_chunks + chunks):
                    c["chunk_index"] = i
                write_list(Path(CHUNKS_JSON), all_chunks + chunks)
                state.retriever = PdfRetriever()     # syncs Chroma: adds new, drops stale

        job.update(
            status="done", stage="Ready", progress=100,
            items=len(items), chunks=len(chunks), sections=len(sections),
        )

    except Exception as e:
        traceback.print_exc()
        job.update(status="error", stage="Failed", error=str(e))


# ============================================================
# API
# ============================================================

app = FastAPI(title="PDF RAG", docs_url="/api/docs", redoc_url=None)


@app.middleware("http")
async def password_gate(request: Request, call_next):
    """HTTP Basic auth, only when APP_PASSWORD is set."""
    if not APP_PASSWORD or request.url.path == "/healthz":
        return await call_next(request)

    header = request.headers.get("authorization", "")
    if header.startswith("Basic "):
        try:
            _, _, password = base64.b64decode(header[6:]).decode("utf-8").partition(":")
            if secrets.compare_digest(password, APP_PASSWORD):
                return await call_next(request)
        except Exception:
            pass

    return Response(
        "Password required", status_code=401,
        headers={"WWW-Authenticate": 'Basic realm="PDF RAG"'},
    )


@app.get("/healthz")
def healthz():
    return {"ok": True}


class AskRequest(BaseModel):
    question: str
    session_id: str
    k: int = 1
    chunk_type: Optional[str] = None


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    name = Path(file.filename or "").name
    if not name.lower().endswith(".pdf"):
        raise HTTPException(400, "Only .pdf files are supported")

    data = await file.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File larger than {MAX_UPLOAD_MB} MB")
    if not data.startswith(b"%PDF"):
        raise HTTPException(400, "File is not a valid PDF")

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = DOCS_DIR / name
    pdf_path.write_bytes(data)

    job_id = uuid.uuid4().hex[:12]
    state.jobs[job_id] = {
        "job_id": job_id, "source": name, "status": "running",
        "stage": "Queued", "progress": 0,
    }
    threading.Thread(target=run_pipeline, args=(job_id, pdf_path), daemon=True).start()
    return state.jobs[job_id]


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    job = state.jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job")
    return job


@app.get("/api/documents")
def documents():
    with state.lock:
        chunks = read_list(Path(CHUNKS_JSON))
        items = read_list(LAYOUT_JSON)

    docs: Dict[str, Dict[str, Any]] = {}
    for c in chunks:
        d = docs.setdefault(c["source"], {"source": c["source"], "chunks": 0, "items": 0, "pages": 0, "types": {}})
        d["chunks"] += 1
        d["pages"] = max(d["pages"], c.get("page_end") or 0)
    for it in items:
        d = docs.get(it.get("source"))
        if d:
            d["items"] += 1
            d["types"][it["type"]] = d["types"].get(it["type"], 0) + 1

    running = [j for j in state.jobs.values() if j["status"] == "running"]
    return {"documents": sorted(docs.values(), key=lambda d: d["source"]), "jobs": running}


@app.get("/api/documents/{source}")
def document_detail(source: str):
    with state.lock:
        items = [x for x in read_list(LAYOUT_JSON) if x.get("source") == source]
        chunks = [c for c in read_list(Path(CHUNKS_JSON)) if c.get("source") == source]
    if not items and not chunks:
        raise HTTPException(404, "Document not found")
    return {"source": source, "items": items, "chunks": chunks}


@app.get("/api/chunks")
def all_chunks():
    """Every chunk of every document, the same JSON that is indexed in Chroma."""
    with state.lock:
        return read_list(Path(CHUNKS_JSON))


@app.post("/api/ask")
def ask(req: AskRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(400, "Question is empty")
    if not state.get_retriever().chunks:
        raise HTTPException(400, "Upload a PDF first")

    answerer = state.get_session(req.session_id)
    print("🚀 ~ ask ~ answerer:", answerer)
    try:
        result = answerer.answer(question, k=req.k, chunk_type=req.chunk_type or None)
    except RuntimeError as e:          # Ollama not running / model not pulled
        raise HTTPException(503, str(e))

    # Attach the retrieved text so the page can show each source
    by_id = answerer.retriever.by_id
    for s in result["sources"]:
        chunk = by_id.get(s["chunk_id"], {})
        s["content"] = chunk.get("content", "")
        s["section_path"] = chunk.get("section_path", "")
    return result


@app.post("/api/reset")
def reset(body: Dict[str, str]):
    answerer = state.sessions.get(body.get("session_id", ""))
    if answerer:
        answerer.reset()
    return {"ok": True}


class FeedbackRequest(BaseModel):
    answer_id: int
    helpful: bool


@app.post("/api/feedback")
def feedback(req: FeedbackRequest):
    """Thumbs up / down on an answer. Enough thumbs-up verifies it; enough thumbs-down removes it from the cache."""
    result = get_cache().vote(req.answer_id, req.helpful)
    if result is None:
        raise HTTPException(404, "Unknown answer")
    return result


# ---------------- static files ----------------

IMAGE_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/images", StaticFiles(directory=IMAGE_DIR), name="images")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


if __name__ == "__main__":
    print(f"\n  Open http://localhost:{PORT}\n")
    uvicorn.run(app, host=HOST, port=PORT)
