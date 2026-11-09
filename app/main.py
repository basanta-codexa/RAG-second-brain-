"""Second Brain test bench — a small, local FastAPI service and vanilla UI.

Purpose: let you upload a PDF/Markdown/text document, merge it with the
existing corpus (``data/``), and check which chunks come back for a question.
Uploading happens entirely on this machine.

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
import tempfile
import threading
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import search as srch
from app import generate
from loader import SUPPORTED_SUFFIXES
from qdrant_store import StoreUnavailable

STATIC_DIR = Path(__file__).resolve().parent / "static"
ALLOWED_EXTENSIONS = frozenset(SUPPORTED_SUFFIXES)
_RESERVED_WINDOWS_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


Mode = Literal["vector", "hybrid", "rerank"]


class SearchRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    top_k: int = Field(default=srch.DEFAULT_TOP_K, ge=1, le=10)
    mode: Mode = "vector"
    # Origin ("data/notes.md") or filename ("notes.md"). Validated against the
    # corpus at query time rather than with a regex that rejects real names.
    source: str | None = Field(default=None, max_length=300)


class AnswerRequest(SearchRequest):
    consent: bool  # must be True; the UI sets it from the session-wide toggle


def sanitize_filename(raw: str) -> str:
    """Reduce a user-supplied name to a safe basename, preserving its suffix.

    The suffix is extracted first so truncation or non-ASCII characters in the
    stem can never turn a supported upload into an invisible file.
    """
    name = Path(raw or "").name            # drop any directory components
    if not name:
        raise HTTPException(status_code=400, detail="the uploaded file has no usable name")

    suffix = Path(name).suffix.lower()
    stem = name[:-len(suffix)] if suffix else name
    stem = stem.replace(" ", "_")
    stem = re.sub(r"[^A-Za-z0-9._-]", "_", stem)  # nothing else survives
    stem = re.sub(r"_+", "_", stem)               # collapse runs
    stem = re.sub(r"_+\.", ".", stem)             # no underscore hugging a dot
    stem = stem.strip("._")

    if not stem:
        if not suffix:
            raise HTTPException(status_code=400, detail="the uploaded file has no usable name")
        stem = "file"

    if stem.split(".")[0].upper() in _RESERVED_WINDOWS_NAMES:
        stem = f"_{stem}"

    max_stem = max(1, 120 - len(suffix))
    stem = stem[:max_stem].rstrip("._") or "file"
    return f"{stem}{suffix}"


def file_info(path: Path, origin: str) -> dict:
    stat = path.stat()
    return {
        "name": path.name,
        "origin": origin,
        "bytes": stat.st_size,
        "modified": stat.st_mtime,
    }


def create_app(data_dir: str | None = None, upload_dir: str | None = None,
               model_name: str | None = None, max_mb: int | None = None,
               chunk_size: int | None = None, overlap: int | None = None) -> FastAPI:
    data_path = Path(data_dir or os.getenv("SECOND_BRAIN_DATA", "data"))
    upload_path = Path(upload_dir or os.getenv("SECOND_BRAIN_UPLOADS", "uploads"))
    model = model_name or os.getenv("SECOND_BRAIN_MODEL", srch.DEFAULT_MODEL)
    max_bytes = (int(max_mb or os.getenv("SECOND_BRAIN_MAX_MB", "25")) * 1024 * 1024)
    chunk_size = int(chunk_size or os.getenv("SECOND_BRAIN_CHUNK_SIZE", srch.DEFAULT_CHUNK_SIZE))
    overlap = int(overlap or os.getenv("SECOND_BRAIN_OVERLAP", srch.DEFAULT_OVERLAP))

    # Loading PyTorch + 400 MB of weights takes ~8s — do it once per process,
    # and reuse it for every query (not every request).
    encoder_lock = threading.Lock()
    encoder_cache: dict = {"instance": None}
    index_lock = threading.Lock()
    index_cache: dict = {"key": None, "index": None}

    def get_encoder():
        with encoder_lock:
            if encoder_cache["instance"] is None:
                encoder_cache["instance"] = srch.load_encoder(model)
            return encoder_cache["instance"]

    data_path.mkdir(parents=True, exist_ok=True)
    upload_path.mkdir(parents=True, exist_ok=True)

    app = FastAPI(title="Second Brain — test bench", version="2.0.0",
                  docs_url=None, redoc_url=None)
    app.state.data_dir = data_path
    app.state.upload_dir = upload_path
    app.state.model = model
    app.state.max_bytes = max_bytes
    app.state.chunk_size = chunk_size
    app.state.overlap = overlap

    def supported_files(folder: Path) -> list[dict]:
        if not folder.is_dir():
            return []
        return [file_info(p, f"{folder.name}/{p.name}") for p in sorted(folder.iterdir())
                if p.is_file() and p.suffix.lower() in ALLOWED_EXTENSIONS]

    def corpus_or_409() -> dict:
        base, uploads = supported_files(data_path), supported_files(upload_path)
        if not base and not uploads:
            raise HTTPException(
                status_code=409,
                detail="no documents yet — upload a PDF/Markdown/text file first, or add files to data/",
            )
        return {"base": base, "uploads": uploads}

    def directory_token(folder: Path) -> tuple:
        return tuple(
            (p.name, p.stat().st_size, p.stat().st_mtime_ns)
            for p in sorted(folder.iterdir())
            if p.is_file() and p.suffix.lower() in ALLOWED_EXTENSIONS
        )

    def current_cache_key() -> tuple:
        return (
            str(data_path), str(upload_path), model, chunk_size, overlap,
            directory_token(data_path), directory_token(upload_path),
        )

    def invalidate_index() -> None:
        index_cache["key"] = None

    def sync_index() -> tuple[dict, dict, bool]:
        """Return (corpus, index, synced_now)."""
        corpus = corpus_or_409()
        key = current_cache_key()
        synced_now = index_cache["key"] != key
        if synced_now:
            index_cache["index"] = srch.create_index(
                [str(data_path), str(upload_path)], model, chunk_size, overlap,
            )
            index_cache["key"] = key
        return corpus, index_cache["index"], synced_now

    def run_search(question: str, top_k: int, mode: Mode, source: str | None
                   ) -> tuple[dict, dict, list[dict], bool]:
        try:
            with index_lock:
                corpus, index, synced_now = sync_index()
                if not index["chunks"]:
                    errors = "; ".join(f"{e['file']}: {e['error']}" for e in index["errors"])
                    raise HTTPException(
                        status_code=422,
                        detail=f"no readable documents could be indexed. {errors}".strip(),
                    )
                encoder = index["encoder"] or get_encoder()
                hits = srch.query_index(
                    index, encoder, question, top_k=top_k, source=source,
                    hybrid=mode == "hybrid", rerank=mode == "rerank",
                )
        except StoreUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except HTTPException:
            raise
        except ImportError as exc:
            raise HTTPException(
                status_code=503,
                detail=f"embedding/reranking dependency missing: {exc}. Install requirements.txt.",
            ) from exc
        except (RuntimeError, OSError, ValueError) as exc:
            raise HTTPException(status_code=503, detail=f"indexing failed: {exc}") from exc
        return corpus, index, hits, synced_now

    # ---------- static UI ----------

    @app.get("/", include_in_schema=False)
    def index_page() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    # ---------- status ----------

    @app.get("/api/status")
    def status() -> dict:
        base, uploads = supported_files(data_path), supported_files(upload_path)
        sources = sorted({f["origin"] for f in base} | {f["origin"] for f in uploads})
        return {
            "base": base,
            "uploads": uploads,
            "documents": len(base) + len(uploads),
            "model": model,
            "max_mb": max_bytes // (1024 * 1024),
            "max_upload_bytes": max_bytes,
            "sources": sources,
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
    async def upload(file: UploadFile = File(...)) -> dict:  # noqa: B008 — FastAPI idiom
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"unsupported file type '{suffix or 'unknown'}'; "
                       f"allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
            )

        safe_name = sanitize_filename(file.filename or "")
        if Path(safe_name).suffix.lower() not in ALLOWED_EXTENSIONS:
            raise HTTPException(status_code=400, detail="unsupported file type after sanitization")

        dest = upload_path / safe_name
        replaced = dest.exists()
        written = 0

        # Write to a temporary file in the same directory, then replace the
        # destination atomically. A rejected upload must never truncate the
        # existing document with the same name.
        fd, tmp_name = tempfile.mkstemp(prefix=".upload-", dir=upload_path)
        try:
            with os.fdopen(fd, "wb") as handle:
                while chunk := await file.read(1024 * 1024):
                    written += len(chunk)
                    if written > max_bytes:
                        raise HTTPException(
                            status_code=413,
                            detail=f"file exceeds the {max_bytes // (1024 * 1024)} MB limit",
                        )
                    handle.write(chunk)
            if written == 0:
                raise HTTPException(status_code=400, detail="empty file rejected")
            os.replace(tmp_name, dest)
        except BaseException:
            try:
                os.unlink(tmp_name)
            except FileNotFoundError:
                pass
            raise

        invalidate_index()
        return {"name": safe_name, "bytes": written, "replaced": replaced}

    @app.delete("/api/uploads/{name}")
    def delete_upload(name: str) -> dict:
        safe = sanitize_filename(name)
        target = upload_path / safe
        if not target.is_file():
            raise HTTPException(status_code=404, detail=f"no such upload: {safe}")
        target.unlink()
        invalidate_index()
        return {"removed": safe}

    @app.post("/api/uploads/clear")
    def clear_uploads() -> dict:
        removed = [p for p in upload_path.iterdir() if p.is_file()]
        for p in removed:
            p.unlink()
        invalidate_index()
        return {"removed": len(removed)}

    # ---------- result shape ----------

    def hit_items(hits: list[dict]) -> list[dict]:
        items = []
        for i, h in enumerate(hits):
            item = {
                "rank": h.get("rank", i + 1),
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

    def warnings(index: dict) -> list[dict]:
        return list(index.get("errors", []))

    # ---------- search ----------

    @app.post("/api/search")
    def search_endpoint(req: SearchRequest) -> dict:
        corpus, index, hits, synced_now = run_search(req.question, req.top_k, req.mode, req.source)
        return {
            "hits": hit_items(hits),
            "cached_index": not synced_now,
            "model": index["model_name"],
            "documents": len(corpus["base"]) + len(corpus["uploads"]),
            "warnings": warnings(index),
        }

    # ---------- grounded answers (Groq) ----------

    @app.post("/api/answer")
    def answer_endpoint(req: AnswerRequest) -> dict:
        if not req.consent:
            raise HTTPException(
                status_code=403,
                detail="generation requires opting in for this session — toggle the allow switch first",
            )

        corpus, index, hits, synced_now = run_search(
            req.question, req.top_k, req.mode, req.source)
        if not hits:
            result = generate.answer(req.question, [])
        else:
            try:
                result = generate.answer(req.question, hits)
            except generate.GenerationError as exc:
                raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
            except ImportError as exc:
                raise HTTPException(
                    status_code=503,
                    detail=f"groq package missing: {exc}. Install requirements.txt.",
                ) from exc

        used = result.pop("used", [])
        return {
            **result,
            "hits": hit_items(used),  # exactly the passages sent, truncated to prompt size
            "cached_index": not synced_now,
            "documents": len(corpus["base"]) + len(corpus["uploads"]),
            "warnings": warnings(index),
        }

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)
