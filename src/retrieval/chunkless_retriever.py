"""Chunkless retriever: BM25 pre-filter + optional reasoning navigator.

Flow:
  1. BM25 over section texts gives a candidate set (top candidate_depth).
  2. Build an outline over candidates (or all sections if small).
  3. If a navigator is provided, ask it to select section indices from the
     outline; map back to section ids.
  4. Otherwise (or on navigator failure), use BM25 top-k.
  5. Return sections in position-aware order (best first AND last to mitigate
     lost-in-the-middle), with full text.

This module operates on in-memory lists (sections + metas) so it stays testable
without the persisted index. The pipeline wires it to the index.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Sequence

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from retriever import BM25  # noqa: E402


def _position_aware_order(items: list, scores: list[float] | None = None) -> list:
    """Reorder so the strongest item is first, second-strongest last, etc.

    Mitigates lost-in-the-middle: the model attends best to the start and end
    of a long context.
    """
    if not items:
        return items
    if scores is None:
        scores = list(range(len(items), 0, -1))
    order = sorted(range(len(items)), key=lambda i: scores[i], reverse=True)
    # interleave: best, worst, 2nd best, 2nd worst, ...
    left, right = [], []
    for rank, i in enumerate(order):
        if rank % 2 == 0:
            left.append(i)
        else:
            right.insert(0, i)
    return [items[i] for i in (left + right)]


class ChunklessRetriever:
    def __init__(self, sections: Sequence[dict], navigator: Callable | None = None,
                 candidate_depth: int = 20, k_default: int = 5):
        """
        sections: list of dicts with keys: id, text, heading, page_start, page_end,
                  summary (optional).
        navigator: callable(outline, query, k=...) -> list[int] (indices into outline)
        """
        self.sections = list(sections)
        self.navigator = navigator
        self.candidate_depth = min(candidate_depth, len(self.sections)) if self.sections else 0
        self.k_default = k_default
        self.bm25 = BM25([s["text"] for s in self.sections]) if self.sections else None

    def search(self, query: str, k: int = None) -> list[dict]:
        k = k or self.k_default
        if not self.sections:
            return []
        # BM25 pre-filter
        if self.bm25 is not None:
            scores = self.bm25.scores(query)
            ranked_idx = sorted(range(len(self.sections)),
                                key=lambda i: scores[i], reverse=True)
            # drop zero-score
            cand = [i for i in ranked_idx if scores[i] > 0][:self.candidate_depth]
            if not cand:
                cand = ranked_idx[:self.candidate_depth]
        else:
            cand = list(range(len(self.sections)))

        outline = []
        for local, gi in enumerate(cand):
            s = self.sections[gi]
            outline.append({
                "idx": local,
                "global": gi,
                "heading": s.get("heading", ""),
                "pages": f"{s.get('page_start','?')}-{s.get('page_end','?')}",
                "summary": s.get("summary", "") or (s["text"][:200]),
            })

        selected_globals = []
        if self.navigator is not None:
            try:
                chosen_locals = self.navigator(outline, query, k=k)
            except Exception:
                chosen_locals = []
            for li in chosen_locals:
                if 0 <= li < len(cand):
                    selected_globals.append(cand[li])
        if not selected_globals:
            # BM25 fallback
            selected_globals = cand[:k]

        # dedupe preserving order
        seen = set()
        uniq = []
        for g in selected_globals:
            if g in seen:
                continue
            seen.add(g)
            uniq.append(g)
        uniq = uniq[:k]

        results = [self.sections[g] for g in uniq]
        # position-aware ordering for generation context
        res_scores = [float(self.bm25.scores(query)[g]) if self.bm25 is not None else 0.0
                      for g in uniq]
        # keep selected order from navigator mostly; apply light reordering only if >2
        if len(results) > 2:
            results = _position_aware_order(results, res_scores)
        return results