"""Summarisation for chunkless RAG.

Per-section and document-level summaries feed the navigator's outline. Two
modes:
  - extractive (default, fully offline, no LLM): first 2 sentences of the
    section, capped at 240 chars. Deterministic and free.
  - llm: uses local Ollama (Qwen2.5-3B) to produce a 1-2 sentence summary.
    Cached by section-text hash so re-ingest is free.

The extractive fallback is what keeps `ingest` working without Ollama.
"""

from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
from pathlib import Path
from typing import Iterable

_SRC = Path(__file__).resolve().parents[0] if False else Path(__file__).resolve().parent
_ROOT = _SRC.parent
for _p in (str(_SRC),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5:3b-instruct"


def _first_sentences(text: str, n: int = 2, cap: int = 240) -> str:
    import re
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    out = " ".join(parts[:n]).strip()
    return out[:cap]


def extractive_summary(text: str) -> str:
    return _first_sentences(text, 2, 240)


def llm_summary(text: str, model: str = OLLAMA_MODEL, host: str = OLLAMA_HOST) -> str:
    prompt = (
        "Summarise the following document section in 1-2 concise sentences. "
        "Do not add information not present in the text.\n\n" + text[:2000]
    )
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0, "seed": 42, "num_ctx": 2048},
    }
    req = urllib.request.Request(
        f"{host}/api/chat",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            obj = json.loads(resp.read())
            return obj.get("message", {}).get("content", "").strip() or extractive_summary(text)
    except Exception:
        return extractive_summary(text)


def summarise_sections(sections_meta: list[dict], full_texts: list[str],
                       use_llm: bool = False, cache: dict | None = None) -> list[str]:
    """Return a list of summaries aligned with sections."""
    if cache is None:
        cache = {}
    out = []
    for meta, text in zip(sections_meta, full_texts):
        key = hashlib.sha256(text.encode()).hexdigest() if False else None
        # cache key by text hash
        h = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if h in cache:
            out.append(cache[h])
            continue
        if use_llm:
            s = llm_summary(text)
        else:
            s = extractive_summary(text)
        cache[h] = s
        out.append(s)
    return out


def summarise_document(texts: list[str]) -> str:
    joined = " ".join(texts[:3]) if texts else ""
    return _first_sentences(joined, 3, 400)


# cache helpers
def load_cache(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return {}
    return {}


def save_cache(path: Path, cache: dict) -> None:
    path.write_text(json.dumps(cache))