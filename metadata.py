"""Local document tags, stored beside the index state.

Tags are metadata only: they never change chunk ids or vectors, so adding or
removing one does not trigger a reindex.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

TAGS_PATH = Path(".cache") / "document_tags.json"
MAX_TAGS = 12
MAX_TAG_LEN = 32
_TAG_RE = re.compile(r"^[a-z0-9][a-z0-9 _-]*$")


def load_tags(path: Path | None = None) -> dict[str, list[str]]:
    target = Path(path) if path is not None else TAGS_PATH
    if not target.exists():
        return {}
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    result: dict[str, list[str]] = {}
    for origin, tags in data.items():
        if isinstance(origin, str) and isinstance(tags, list):
            result[origin] = [t for t in tags
                              if isinstance(t, str) and _TAG_RE.fullmatch(t)]
    return result


def normalize_tags(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if not isinstance(value, str):
            continue
        tag = re.sub(r"\s+", " ", value.strip().lower()).strip(" -_")
        if not tag or len(tag) > MAX_TAG_LEN or not _TAG_RE.fullmatch(tag):
            continue
        if tag not in result:
            result.append(tag)
        if len(result) >= MAX_TAGS:
            break
    return result


def save_tags(tags: dict[str, list[str]], path: Path | None = None) -> None:
    target = Path(path) if path is not None else TAGS_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(tags, indent=2), encoding="utf-8")
    tmp.replace(target)
