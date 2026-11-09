"""Reasoning-based navigator for chunkless RAG.

Given a document outline (list of {idx, heading, pages, summary}) and a query,
ask the local LLM which sections to read, returning an ordered list of section
indices. Two strategies:
  - batch: present the whole outline, ask for top-k sections (flat).
  - tree:  ask the LLM to walk the heading hierarchy top-down.

Output is validated against the outline; on any parse failure or unknown id the
navigator returns an empty list so the caller falls back to BM25. This keeps the
offline path robust.
"""

from __future__ import annotations

import json
import re
import sys
import urllib.request
from pathlib import Path
from typing import Sequence

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5:3b-instruct"


def _chat(prompt: str, model: str = OLLAMA_MODEL, host: str = OLLAMA_HOST,
          num_ctx: int = 4096) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0, "seed": 42, "num_ctx": num_ctx},
    }
    req = urllib.request.Request(
        f"{host}/api/chat",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        obj = json.loads(resp.read())
    return obj.get("message", {}).get("content", "")


def _extract_json(text: str):
    # Find the first JSON array or object in the response.
    m = re.search(r"\[[\s\S]*\]|\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def navigate_batch(outline: Sequence[dict], query: str, k: int = 5,
                   model: str = OLLAMA_MODEL, host: str = OLLAMA_HOST) -> list[int]:
    """Flat batch selection: return up to k section indices."""
    lines = []
    for o in outline:
        lines.append(f"[{o['idx']}] {o['heading']} (pp.{o['pages']}): {o.get('summary','')}")
    outline_text = "\n".join(lines)
    prompt = (
        "You are selecting document sections to answer a question.\n"
        "Given the outline below, return a JSON array of section indices (integers) "
        "that are most likely to contain the answer, in priority order. "
        "Return at most %d indices. Return ONLY the JSON array, no prose.\n\n"
        "Outline:\n%s\n\nQuestion: %s\n" % (k, outline_text, query)
    )
    try:
        raw = _chat(prompt, model=model, host=host)
    except Exception:
        return []
    parsed = _extract_json(raw)
    if not isinstance(parsed, list):
        return []
    valid = {o["idx"] for o in outline}
    out = []
    for x in parsed:
        try:
            i = int(x)
        except (TypeError, ValueError):
            continue
        if i in valid and i not in out:
            out.append(i)
        if len(out) >= k:
            break
    return out


def navigate_tree(outline: Sequence[dict], query: str, k: int = 5,
                  model: str = OLLAMA_MODEL, host: str = OLLAMA_HOST) -> list[int]:
    """Top-down traversal: ask LLM to pick a branch, then drill. Simplified:
    ask for top-k with a hint to prefer parent sections first."""
    # For simplicity, reuse batch but request hierarchical reasoning in prompt.
    lines = []
    for o in outline:
        lines.append(f"[{o['idx']}] {o['heading']} (pp.{o['pages']}): {o.get('summary','')}")
    outline_text = "\n".join(lines)
    prompt = (
        "You are navigating a document's table of contents top-down to answer a question.\n"
        "First identify the most relevant top-level section, then any subsections.\n"
        "Return a JSON array of section indices in the order you would read them, "
        "at most %d. Return ONLY the JSON array.\n\n"
        "Outline:\n%s\n\nQuestion: %s\n" % (k, outline_text, query)
    )
    try:
        raw = _chat(prompt, model=model, host=host)
    except Exception:
        return []
    parsed = _extract_json(raw)
    if not isinstance(parsed, list):
        return []
    valid = {o["idx"] for o in outline}
    out = []
    for x in parsed:
        try:
            i = int(x)
        except (TypeError, ValueError):
            continue
        if i in valid and i not in out:
            out.append(i)
        if len(out) >= k:
            break
    return out


def navigate(outline, query, strategy: str = "batch", k: int = 5, **kw) -> list[int]:
    if strategy == "tree":
        return navigate_tree(outline, query, k=k, **kw)
    return navigate_batch(outline, query, k=k, **kw)