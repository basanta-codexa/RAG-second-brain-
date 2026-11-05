"""Second Brain test bench — a small, local FastAPI service and vanilla UI.

Purpose: let you upload a PDF/Markdown/text document, merge it with the
existing corpus (``data/``), and check which chunks come back for a question.
Uploading happens entirely on this machine. The service does not generate
answers; it shows the evidence.

Run:
    uvicorn app.main:app --reload
    # or: python -m app.main
    # then open http://127.0.0.1:8000

Configuration (environment variables):
    SECOND_BRAIN_DATA     base corpus folder (default: data)
    SECOND_BRAIN_UPLOADS  upload folder      (default: uploads)
    SECOND_BRAIN_MODEL    embedding model    (default: BAAI/bge-base-en-v1.5)
    SECOND_BRAIN_MAX_MB   max upload size    (default: 25 MB)
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import search as srch
from loader import SUPPORTED_SUFFIXES
from app import generate

STATIC_DIR = Path(__file__).resolve().parent / "static"
ALLOWED_EXTENSIONS = frozenset(SUPPORTED_SUFFIXES)


Mode = Literal["vector", "hybrid", "rerank"]


class SearchRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    top_k: int = Field(default=srch.DEFAULT_TOP_K, ge=1, le=10)
    mode: Mode = "vector"
    source: str | None = Field(default=None, max_length=120,
                               pattern=r"^[A-Za-z0-9][A-Za-z0-9._ -]{0,119}$")


class AnswerRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    top_k: int = Field(default=srch.DEFAULT_TOP_K, ge=1, le=10)
    consent: bool  # must be True; the UI sets it from the session-wide toggle
    mode: Mode = "vector"
    source: str | None = Field(default=None, max_length=120,
                               pattern=r"^[A-Za-z0-9][A-Za-z0-9._ -]{0,119}$")


def sanitize_filename(raw: str) -> str:
    """Reduce a user-supplied name to a safe basename inside the upload dir."""
    name = Path(raw or "").name            # drop any directory components
    name = name.replace(" ", "_")
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name)  # nothing else survives
    name = re.sub(r"_+", "_", name)               # collapse runs
    name = re.sub(r"_+\.", ".", name)             # no underscore hugging the extension
    name = name.strip("._")                       # no hidden files / dotfile games
    if not name:
        raise HTTPException(status_code=400, detail="the uploaded file has no usable name")
    return name[:120]


def file_info(path: Path, source: str) -> dict:
    stat = path.stat()
    return {
        "name": path.name,
        "bytes": stat.st_size,
        "modified": stat.st_mtime,
        "kind": source,
    }


def create_app(data_dir: str | None = None, upload_dir: str | None = None,
               model_name: str | None = None, max_mb: int | None = None) -> FastAPI:
    data_path = Path(data_dir or os.getenv("SECOND_BRAIN_DATA", "data"))
    upload_path = Path(upload_dir or os.getenv("SECOND_BRAIN_UPLOADS", "uploads"))
    model = model_name or os.getenv("SECOND_BRAIN_MODEL", srch.DEFAULT_MODEL)
    max_bytes = (int(max_mb or os.getenv("SECOND_BRAIN_MAX_MB", "25")) * 1024 * 1024)

    # Loading PyTorch + 400 MB of weights takes ~8s — do it once per process,
    # and reuse it for every query (not every request).
    _encoder_cache: dict = {"instance": None}

    def get_encoder() -> srch.SentenceTransformer:
        if _encoder_cache["instance"] is None:
            _encoder_cache["instance"] = srch.SentenceTransformer(model)
        return _encoder_cache["instance"]

    data_path.mkdir(parents=True, exist_ok=True)
    upload_path.mkdir(parents=True, exist_ok=True)

    app = FastAPI(title="Second Brain — test bench", version="1.0.0", docs_url=None, redoc_url=None)
    app.state.data_dir = data_path
    app.state.upload_dir = upload_path
    app.state.model = model
    app.state.max_bytes = max_bytes

    def supported_files(folder: Path) -> list[dict]:
        if not folder.is_dir():
            return []
        return [file_info(p, folder.name) for p in sorted(folder.iterdir())
                if p.is_file() and p.suffix.lower() in ALLOWED_EXTENSIONS]

    def corpus_or_400() -> dict:
        base, uploads = supported_files(data_path), supported_files(upload_path)
        if not base and not uploads:
            raise HTTPException(
                status_code=409,
                detail="no documents yet — upload a PDF/Markdown/text file first, or add files to data/",
            )
        return {"base": base, "uploads": uploads}

    # ---------- static UI ----------

    @app.get("/", include_in_schema=False)
    def index_page() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    # ---------- status ----------

    @app.get("/api/status")
    def status() -> dict:
        base, uploads = supported_files(data_path), supported_files(upload_path)
        names = sorted({f["name"] for f in base} | {f["name"] for f in uploads})
        return {
            "base": [f for f in base],
            "uploads": [f for f in uploads],
            "documents": len(base) + len(uploads),
            "model": model,
            "max_mb": max_bytes // (1024 * 1024),
            "max_upload_bytes": max_bytes,
            "sources": names,
        }

    @app.get("/api/index-status")
    def index_status() -> dict:
        progress_file = srch.PROGRESS_PATH
        if not progress_file.exists():
            return {"active": False}
        try:
            return json.loads(progress_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"active": False}

    # ---------- upload / clear ----------

    @app.post("/api/upload")
    async def upload(file: UploadFile = File(...)) -> dict:
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            raise HTTPException(status_code=400, detail=f"unsupported file type '{suffix or 'unknown'}'; allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}")

        safe_name = sanitize_filename(file.filename or "")
        dest = upload_path / safe_name
        replaced = dest.exists()

        written = 0
        try:
            with dest.open("wb") as handle:
                while chunk := await file.read(1024 * 1024):
                    written += len(chunk)
                    if written > max_bytes:
                        raise HTTPException(
                            status_code=413,
                            detail=f"file exceeds the {app.state.max_bytes // (1024 * 1024)} MB limit",
                        )
                    handle.write(chunk)
        except BaseException:
            # never leave a half-written or oversized upload behind
            dest.unlink(missing_ok=True)
            raise

        if written == 0:
            dest.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail="empty file rejected")

        return {"name": safe_name, "bytes": written, "replaced": replaced}

    @app.delete("/api/uploads/{name}")
    def delete_upload(name: str) -> dict:
        safe = sanitize_filename(name)
        target = upload_path / safe
        if not target.is_file():
            raise HTTPException(status_code=404, detail=f"no such upload: {safe}")
        target.unlink()
        return {"removed": safe}

    @app.post("/api/uploads/clear")
    def clear_uploads() -> dict:
        removed = [p for p in upload_path.iterdir() if p.is_file()]
        for p in removed:
            p.unlink()
        return {"removed": len(removed)}

    def build_index_or_fail():
        corpus = corpus_or_400()
        index = srch.create_index([str(data_path), str(upload_path)], model)
        encoder = index["encoder"] or get_encoder()
        return corpus, index, encoder

    def mode_kwargs(mode: Mode) -> dict:
        return {"hybrid": mode == "hybrid", "rerank": mode == "rerank"}

    def hit_items(hits: list[dict]) -> list[dict]:
        items = []
        for i, h in enumerate(hits):
            item = {
                "rank": i + 1,
                "score": round(h["score"], 4),
                "source": h["source"],
                "page": h["page"],
                "chunk_id": h["chunk_id"],
                "origin": h.get("origin", ""),
                "text": h["text"],
            }
            for extra in ("bm25_score", "rerank_score"):
                if extra in h:
                    item[extra] = round(float(h[extra]), 4)
            items.append(item)
        return items

    # ---------- search ----------

    @app.post("/api/search")
    def search_endpoint(req: SearchRequest) -> dict:
        corpus, index, encoder = build_index_or_fail()
        hits = srch.query_index(index, encoder, req.question, top_k=req.top_k,
                                source=req.source, **mode_kwargs(req.mode))

        items = hit_items(hits)
        return {
            "hits": items,
            "cached_index": index["encoder"] is None,
            "model": index["model_name"],
            "documents": len(corpus["base"]) + len(corpus["uploads"]),
        }

    # ---------- grounded answers (Groq) ----------

    @app.post("/api/answer")
    def answer_endpoint(req: AnswerRequest) -> dict:
        if not req.consent:
            raise HTTPException(
                status_code=403,
                detail="generation requires opting in for this session — toggle the allow switch first",
            )

        corpus, index, encoder = build_index_or_fail()
        hits = srch.query_index(index, encoder, req.question, top_k=req.top_k,
                                source=req.source, **mode_kwargs(req.mode))

        try:
            result = generate.answer(req.question, hits)
        except RuntimeError as exc:
            msg = str(exc)
            raise HTTPException(status_code=503 if "GROQ_API_KEY" in msg else 502, detail=msg)

        items = hit_items(hits)
        return {
            **result,
            "hits": items,  # same shape as /api/search — UI renders these as the evidence list
            "cached_index": index["encoder"] is None,
            "documents": len(corpus["base"]) + len(corpus["uploads"]),
        }

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=False)
