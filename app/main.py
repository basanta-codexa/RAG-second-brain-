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
    SECOND_BRAIN_RERANKER cross-encoder model for rerank modes
    SECOND_BRAIN_MAX_MB   max upload size    (default: 25 MB)
"""
from __future__ import annotations

import importlib.util
import json
import logging
import os
import re
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from filelock import Timeout
from pydantic import BaseModel, Field, field_validator

import search as srch
from app import generate
from chunker import chunk_records
from loader import SUPPORTED_SUFFIXES, load_pdf, load_text
from metadata import (
    COLLECTIONS_PATH,
    TAGS_PATH,
    load_collections,
    load_tags,
    normalize_tags,
    save_collections,
    save_tags,
)
from qdrant_store import Store, StoreUnavailable, load_state_file, sha256_file

logger = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).resolve().parent / "static"
# Loaded explicitly here rather than relying on app.generate's import side
# effect, so every SECOND_BRAIN_* setting is available when create_app() runs.
load_dotenv(STATIC_DIR.parent.parent / ".env")
ALLOWED_EXTENSIONS = frozenset(SUPPORTED_SUFFIXES)
_RESERVED_WINDOWS_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


Mode = Literal["vector", "hybrid", "rerank", "hybrid+rerank"]

# A web request should fail fast when the CLI (or another process) is syncing,
# instead of holding a worker thread for the full CLI lock timeout.
WEB_LOCK_TIMEOUT = 5.0


class SearchRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    top_k: int = Field(default=srch.DEFAULT_TOP_K, ge=1, le=10)
    mode: Mode = "vector"
    # Origin ("data/notes.md") or filename ("notes.md"). Validated against the
    # corpus at query time rather than with a regex that rejects real names.
    source: str | None = Field(default=None, max_length=300)

    @field_validator("question")
    @classmethod
    def _non_blank_question(cls, value: str) -> str:
        value = value.strip()
        if len(value) < 3:
            raise ValueError("question must be at least 3 non-whitespace characters")
        return value


class AnswerRequest(SearchRequest):
    consent: bool  # must be True; the UI sets it from the session-wide toggle


class TagsRequest(BaseModel):
    tags: list[str] = Field(default_factory=list, max_length=12)


class CollectionRequest(BaseModel):
    collection: str | None = None


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
               chunk_size: int | None = None, overlap: int | None = None,
               reranker_model: str | None = None,
               chunk_mode: str | None = None,
               token_size: int | None = None,
               token_overlap: int | None = None) -> FastAPI:
    data_path = Path(data_dir or os.getenv("SECOND_BRAIN_DATA", "data"))
    upload_path = Path(upload_dir or os.getenv("SECOND_BRAIN_UPLOADS", "uploads"))
    model = model_name or os.getenv("SECOND_BRAIN_MODEL", srch.DEFAULT_MODEL)
    reranker = reranker_model or os.getenv("SECOND_BRAIN_RERANKER") or srch.RERANK_MODEL
    max_bytes = (int(max_mb or os.getenv("SECOND_BRAIN_MAX_MB", "25")) * 1024 * 1024)
    chunk_size = int(chunk_size or os.getenv("SECOND_BRAIN_CHUNK_SIZE", srch.DEFAULT_CHUNK_SIZE))
    overlap = int(overlap or os.getenv("SECOND_BRAIN_OVERLAP", srch.DEFAULT_OVERLAP))
    tags_path = Path(os.getenv("SECOND_BRAIN_TAGS", str(TAGS_PATH)))
    collections_path = Path(os.getenv("SECOND_BRAIN_COLLECTIONS", str(COLLECTIONS_PATH)))
    chunk_mode = chunk_mode or os.getenv("SECOND_BRAIN_CHUNK_MODE", srch.DEFAULT_CHUNK_MODE)
    token_size = int(token_size or os.getenv("SECOND_BRAIN_TOKEN_SIZE", srch.DEFAULT_TOKEN_SIZE))
    token_overlap = int(
        token_overlap or os.getenv("SECOND_BRAIN_TOKEN_OVERLAP", srch.DEFAULT_TOKEN_OVERLAP))

    # Loading PyTorch + 400 MB of weights takes ~8s — do it once per process,
    # and reuse it for every query (not every request).
    encoder_lock = threading.Lock()
    encoder_cache: dict = {"instance": None}
    state_lock = threading.Lock()
    index_cache: dict = {"key": None, "index": None, "encoder": None}

    def get_encoder():
        with encoder_lock:
            if encoder_cache["instance"] is None:
                encoder_cache["instance"] = srch.load_encoder(model)
            return encoder_cache["instance"]

    data_path.mkdir(parents=True, exist_ok=True)
    upload_path.mkdir(parents=True, exist_ok=True)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Torch + model loading costs ~25 s on a cold process. Do it while the
        # app boots rather than on the user's first question.
        if importlib.util.find_spec("sentence_transformers") is not None:
            threading.Thread(target=get_encoder, daemon=True,
                             name="encoder-warmup").start()
        yield

    app = FastAPI(title="Second Brain — Knowledge Workspace", version="2.0.0",
                  lifespan=lifespan)

    @app.middleware("http")
    async def log_requests(request, call_next):
        started = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - started) * 1000
        response.headers["X-Response-Time-Ms"] = f"{elapsed_ms:.1f}"
        logger.info("%s %s -> %s in %.1f ms", request.method, request.url.path,
                    response.status_code, elapsed_ms)
        return response

    app.state.data_dir = data_path
    app.state.upload_dir = upload_path
    app.state.model = model
    app.state.reranker_model = reranker
    app.state.max_bytes = max_bytes
    app.state.chunk_size = chunk_size
    app.state.overlap = overlap

    def supported_files(folder: Path) -> list[dict]:
        if not folder.is_dir():
            return []
        return [file_info(p, f"{folder.name}/{p.name}") for p in sorted(folder.iterdir())
                if p.is_file() and p.suffix.lower() in ALLOWED_EXTENSIONS]

    def all_document_paths() -> list[tuple[Path, str]]:
        """Every supported file with its origin, across both folders."""
        found = []
        for folder in (data_path, upload_path):
            if not folder.is_dir():
                continue
            for path in sorted(folder.iterdir()):
                if path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS:
                    found.append((path, f"{folder.name}/{path.name}"))
        return found

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
        config = (str(data_path), str(upload_path), model, chunk_size,
                  overlap, chunk_mode)
        if chunk_mode == "token":
            config += (token_size, token_overlap)
        return config + (directory_token(data_path), directory_token(upload_path))

    def invalidate_index() -> None:
        index_cache["key"] = None

    def purge_upload_vectors(names: list[str]) -> None:
        """Remove Qdrant points for uploads before their files are unlinked."""
        if not names:
            return
        try:
            store = Store()
            with store.lock(timeout=WEB_LOCK_TIMEOUT):
                for name in names:
                    store.delete_by_origin(f"{upload_path.name}/{name}")
        except Timeout as exc:
            raise HTTPException(
                status_code=409,
                detail="another index sync is running; retry shortly",
                headers={"Retry-After": "5"},
            ) from exc
        except StoreUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    def sync_index() -> tuple[dict, dict, bool]:
        """Return (corpus, index, synced_now)."""
        corpus = corpus_or_409()
        key = current_cache_key()
        synced_now = index_cache["key"] != key
        if synced_now:
            index = srch.create_index(
                [str(data_path), str(upload_path)], model, chunk_size, overlap,
                reranker_model=reranker, lock_timeout=WEB_LOCK_TIMEOUT,
                encoder=index_cache["encoder"],
                chunk_mode=chunk_mode, token_size=token_size,
                token_overlap=token_overlap,
            )
            # create_index returns the encoder it used; reuse it on the next
            # sync instead of loading the model a second time per request.
            index_cache["encoder"] = index.get("encoder")
            index_cache["index"] = index
            # A sync with file errors is not a stable state: keep the key unset
            # so the next request retries instead of caching a partial index.
            cacheable = not index.get("errors")
            index_cache["key"] = key if cacheable else None
            if not cacheable:
                logger.warning("index sync reported %d file error(s); retrying on next request",
                               len(index["errors"]))
            else:
                logger.info("index synced: %d chunks from %d document(s)",
                            len(index["chunks"]),
                            len(corpus["base"]) + len(corpus["uploads"]))
        return corpus, index_cache["index"], synced_now

    def run_search(question: str, top_k: int, mode: Mode, source: str | None
                   ) -> tuple[dict, dict, list[dict], bool]:
        try:
            with state_lock:
                corpus, index, synced_now = sync_index()
                if not index["chunks"]:
                    errors = "; ".join(f"{e['file']}: {e['error']}" for e in index["errors"])
                    raise HTTPException(
                        status_code=422,
                        detail=f"no readable documents could be indexed. {errors}".strip(),
                    )
                if source:
                    known = {c.get("origin") for c in index["chunks"]}
                    known |= {c["source"] for c in index["chunks"]}
                    if source not in known:
                        raise HTTPException(status_code=400,
                                            detail=f"unknown source: {source}")
                encoder = index["encoder"] or get_encoder()
                hits = srch.query_index(
                    index, encoder, question, top_k=top_k, source=source,
                    hybrid=mode in ("hybrid", "hybrid+rerank"),
                    rerank=mode in ("rerank", "hybrid+rerank"),
                )
        except srch.IndexBusy as exc:
            raise HTTPException(
                status_code=409,
                detail=str(exc),
                headers={"Retry-After": "5"},
            ) from exc
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
            "collections": load_collections(collections_path),
        }

    @app.get("/api/documents/{folder}/{name}")
    def document_detail(folder: str, name: str) -> dict:
        """Return one document's metadata, page/chunk counts and a short preview."""
        if folder == data_path.name:
            base = data_path
        elif folder == upload_path.name:
            base = upload_path
        else:
            raise HTTPException(status_code=404, detail=f"unknown folder: {folder}")
        if Path(name).name != name:
            raise HTTPException(status_code=400, detail="name must be a plain filename")
        target = base / name
        if not target.is_file() or target.suffix.lower() not in ALLOWED_EXTENSIONS:
            raise HTTPException(status_code=404, detail=f"no such document: {folder}/{name}")

        try:
            records = load_pdf(target) if target.suffix.lower() == ".pdf" else load_text(target)
            chunks = chunk_records(records, chunk_size, overlap)
            digest = sha256_file(target)
        except (OSError, ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=422, detail=f"could not read document: {exc}") from exc

        origin = f"{folder}/{name}"
        tags = load_tags(tags_path).get(origin, [])
        collection = load_collections(collections_path).get(origin)
        cached = index_cache["index"] or {}
        state_entry = (load_state_file() or {}).get("files", {}).get(origin)
        index_errors = [e for e in cached.get("errors", []) if e.get("file") == origin]
        if index_errors:
            status = "error"
        elif state_entry is None:
            status = "not indexed"
        elif state_entry.get("sha256") == digest:
            status = "indexed"
        else:
            status = "stale — will re-embed on the next search"
        preview = "\n\n".join(f"<<<PAGE {r['page']}>>>\n{r['text']}" for r in records[:2])

        duplicates = []
        for other, other_origin in all_document_paths():
            if other == target:
                continue
            try:
                if sha256_file(other) == digest:
                    duplicates.append(other_origin)
            except OSError:
                continue

        return {
            "name": name,
            "origin": origin,
            "bytes": target.stat().st_size,
            "modified": target.stat().st_mtime,
            "sha256": digest,
            "tags": tags,
            "collection": collection,
            "duplicates": duplicates,
            "pages": len(records),
            "chunks": len(chunks),
            "preview": preview[:2000],
            "indexed": any(c.get("origin") == origin for c in cached.get("chunks", [])),
            "status": status,
            "index_errors": index_errors,
        }

    @app.put("/api/documents/{folder}/{name}/collection")
    def set_document_collection(folder: str, name: str,
                                body: CollectionRequest) -> dict:
        """Assign a document to a named collection (``None`` clears it)."""
        if folder == data_path.name:
            base = data_path
        elif folder == upload_path.name:
            base = upload_path
        else:
            raise HTTPException(status_code=404, detail=f"unknown folder: {folder}")
        if Path(name).name != name:
            raise HTTPException(status_code=400, detail="name must be a plain filename")
        if not (base / name).is_file():
            raise HTTPException(status_code=404, detail=f"no such document: {folder}/{name}")

        origin = f"{folder}/{name}"
        all_collections = load_collections(collections_path)
        names = normalize_tags([body.collection or ""])
        if names:
            all_collections[origin] = names[0]
        else:
            all_collections.pop(origin, None)
        save_collections(all_collections, collections_path)
        return {"origin": origin, "collection": all_collections.get(origin)}

    @app.put("/api/documents/{folder}/{name}/tags")
    def set_document_tags(folder: str, name: str, body: TagsRequest) -> dict:
        """Replace the tags for one document. Does not affect the index."""
        if folder == data_path.name:
            base = data_path
        elif folder == upload_path.name:
            base = upload_path
        else:
            raise HTTPException(status_code=404, detail=f"unknown folder: {folder}")
        if Path(name).name != name:
            raise HTTPException(status_code=400, detail="name must be a plain filename")
        if not (base / name).is_file():
            raise HTTPException(status_code=404, detail=f"no such document: {folder}/{name}")

        origin = f"{folder}/{name}"
        all_tags = load_tags(tags_path)
        all_tags[origin] = normalize_tags(body.tags)
        save_tags(all_tags, tags_path)
        return {"origin": origin, "tags": all_tags[origin]}

    @app.get("/api/documents/{folder}/{name}/pages/{page}")
    def document_page(folder: str, name: str, page: int) -> Response:
        """Render one PDF page as PNG for the detail dialog."""
        if folder == data_path.name:
            base = data_path
        elif folder == upload_path.name:
            base = upload_path
        else:
            raise HTTPException(status_code=404, detail=f"unknown folder: {folder}")
        if Path(name).name != name or page < 1:
            raise HTTPException(status_code=400, detail="invalid document or page number")
        target = base / name
        if not target.is_file() or target.suffix.lower() != ".pdf":
            raise HTTPException(status_code=404,
                                detail="page previews are available for PDFs only")

        try:
            import pymupdf as fitz

            with fitz.open(str(target)) as doc:
                if page > len(doc):
                    raise HTTPException(status_code=404,
                                        detail=f"document has {len(doc)} page(s)")
                pix = doc.load_page(page - 1).get_pixmap(
                    matrix=fitz.Matrix(1.5, 1.5), alpha=False)
                png = pix.tobytes("png")
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=422,
                                detail=f"could not render page: {exc}") from exc

        return Response(content=png, media_type="image/png",
                        headers={"Cache-Control": "public, max-age=3600"})

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

    def _save_upload(file: UploadFile) -> dict:
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
                while chunk := file.file.read(1024 * 1024):
                    written += len(chunk)
                    if written > max_bytes:
                        raise HTTPException(
                            status_code=413,
                            detail=f"file exceeds the {max_bytes // (1024 * 1024)} MB limit",
                        )
                    handle.write(chunk)
            if written == 0:
                raise HTTPException(status_code=400, detail="empty file rejected")
            # The visible swap and cache invalidation share the lock used by
            # run_search, so a concurrent query cannot observe a half-updated
            # directory (directory_token would stat a file being replaced).
            with state_lock:
                os.replace(tmp_name, dest)
                invalidate_index()
        except BaseException:
            try:
                os.unlink(tmp_name)
            except FileNotFoundError:
                pass
            raise

        return {"name": safe_name, "bytes": written, "replaced": replaced}

    @app.post("/api/upload")
    def upload(
        file: UploadFile | None = File(default=None),  # noqa: B008 — FastAPI idiom
        files: list[UploadFile] | None = File(default=None),  # noqa: B008
    ) -> dict:
        """Accept one or more documents. ``file`` is kept for older clients."""
        incoming = ([file] if file is not None else []) + list(files or [])
        if not incoming:
            raise HTTPException(status_code=400, detail="no files supplied")

        saved: list[dict] = []
        errors: list[dict] = []
        for item in incoming:
            try:
                saved.append(_save_upload(item))
            except HTTPException as exc:
                errors.append({
                    "file": item.filename or "unknown",
                    "error": exc.detail,
                    "status": exc.status_code,
                })
        if not saved and errors:
            raise HTTPException(status_code=errors[0]["status"], detail=errors[0]["error"])

        return {
            **(saved[0] if saved else {}),
            "saved": saved,
            "errors": errors,
            "count": len(saved),
        }

    @app.delete("/api/uploads/{name}")
    def delete_upload(name: str) -> dict:
        safe = sanitize_filename(name)
        with state_lock:
            target = upload_path / safe
            if not target.is_file():
                raise HTTPException(status_code=404, detail=f"no such upload: {safe}")
            purge_upload_vectors([safe])
            target.unlink()
            invalidate_index()
        return {"removed": safe}

    @app.post("/api/uploads/clear")
    def clear_uploads() -> dict:
        with state_lock:
            # Skip in-flight .upload-* temp files from a concurrent upload.
            removed = [p for p in upload_path.iterdir()
                       if p.is_file() and not p.name.startswith(".upload-")]
            purge_upload_vectors([p.name for p in removed])
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
