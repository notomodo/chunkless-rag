"""Structure extraction for chunkless RAG.

Instead of flattening a PDF into fixed-size chunks, this module builds a
`DocumentTree`: a hierarchy of sections (and, where present, their numbered
clauses) that preserves the document's native organisation. The tree is what
the chunkless retriever navigates and what the navigator LLM reasons over.

Design notes
------------
- Reuses `ingest_sme.extract_lines` for page-furniture removal and reading
  order. Does NOT reuse the chunker: that is the whole point.
- Section splitting reuses the same `SECTION_RE` heuristic as `ingest_sme`
  (integer + dot + capitalised title), so behaviour stays consistent with the
  baseline's section detection, but the body is KEPT WHOLE instead of being
  packed to a 400-token budget.
- Pages with no detectable heading become synthetic `Section (page N)` units,
  so every page stays reachable (no orphaned content).
- Clause splitting uses `CLAUSE_RE` (numbered clause markers) to provide a
  second, finer level of the tree for drill-down.
"""

from __future__ import annotations

import hashlib
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ingest_sme import (  # noqa: E402
    BULLET_RE,
    CLAUSE_RE,
    SECTION_RE,
    extract_lines,
    is_heading,
)


@dataclass
class Clause:
    """A numbered clause inside a section."""
    clause_id: str            # e.g. "12.3"
    text: str
    page: int
    char_start: int
    char_end: int


@dataclass
class Section:
    """A top-level section of a document."""
    section_id: str           # assigned later as f"{doc_sha[:8]}:s{ordinal}"
    heading: str
    level: int
    page_start: int
    page_end: int
    text: str                 # heading + body (whole, unchunked)
    char_start: int
    char_end: int
    clauses: list[Clause] = field(default_factory=list)
    summary: str = ""         # filled by summariser

    @property
    def title(self) -> str:
        return self.heading.strip() or f"Section (p{self.page_start})"


@dataclass
class DocumentTree:
    sha256: str
    filename: str
    pages: int
    sections: list[Section]

    def outline(self) -> list[dict]:
        """Compact outline for the navigator LLM."""
        out = []
        for i, s in enumerate(self.sections):
            out.append({
                "idx": i,
                "heading": s.heading,
                "pages": f"{s.page_start}-{s.page_end}",
                "summary": s.summary or _first_sentences(s.text, 2),
                "n_chars": len(s.text),
            })
        return out


def _first_sentences(text: str, n: int = 2) -> str:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return " ".join(parts[:n])[:400]


def _page_at(offsets, idx: int) -> int:
    best = offsets[0][1] if offsets else 1
    for start, pno in offsets:
        if start <= idx:
            best = pno
        else:
            break
    return best


def build_tree(path: Path, sha256: str | None = None) -> DocumentTree:
    """Build a DocumentTree from a PDF path."""
    if sha256 is None:
        sha256 = hashlib.sha256(path.read_bytes()).hexdigest()

    lines = extract_lines(path)
    if not lines:
        return DocumentTree(sha256=sha256, filename=path.name, pages=0, sections=[])

    pages = max(pno for pno, _ in lines)

    # Join into a stream (same as ingest_sme PASS 1).
    parts: list[str] = []
    page_of: list[int] = []
    for pno, line in lines:
        parts.append(line)
        page_of.append(pno)
    stream = " ".join(parts)

    offsets: list[tuple[int, int]] = []
    pos = 0
    for line, pno in zip(parts, page_of):
        offsets.append((pos, pno))
        pos += len(line) + 1

    # Segmentation strategy (hybrid):
    # 1. If the document has detectable headings (short title/question lines),
    #    segment by headings: each heading starts a new section and owns the
    #    following non-heading lines. This preserves FAQ/policy structure.
    # 2. Else fall back to the inline numbered-section markers (SECTION_RE).
    # 3. Else treat the whole document as one section.
    heading_lines = [(i, ln) for i, (pn, ln) in enumerate(lines) if is_heading(ln)]

    raw_segments: list[tuple[str, str, int, int]] = []  # heading, body, page_start, char_start

    if heading_lines:
        # Build sections by grouping consecutive lines under each heading.
        # A heading line belongs to the NEW section; body = lines after it
        # until the next heading.
        cur_head: str | None = None
        cur_body: list[str] = []
        cur_page: int = lines[0][0]
        cur_start: int = 0
        # char offset tracking via stream positions is complex; use 0-based
        # approximate offsets by locating body in stream later. We'll set
        # char_start=0 for heading-based segments (offsets still monotonic
        # enough for citations; faithful offsets are refined below).
        segments_pending: list[tuple[str, list[tuple[int, str]], int]] = []
        # group
        groups: list[tuple[str, list[tuple[int, str]]]] = []
        current_heading = path.stem.replace("_", " ")
        current_body: list[tuple[int, str]] = []
        for pn, ln in lines:
            if is_heading(ln) and not BULLET_RE.match(ln) and ln.strip()[:1] not in "•-●*–▪◦":
                if current_body or (current_heading and current_heading != ln):
                    groups.append((current_heading, current_body))
                current_heading = ln
                current_body = []
            else:
                current_body.append((pn, ln))
        if current_body or current_heading:
            groups.append((current_heading, current_body))

        # Collapse duplicate consecutive headings and drop empties.
        cleaned: list[tuple[str, list[tuple[int, str]]]] = []
        for head, body_lines in groups:
            if not head.strip() and not body_lines:
                continue
            if cleaned and cleaned[-1][0] == head and not body_lines:
                continue
            cleaned.append((head, body_lines))
        groups = cleaned

        # Convert groups to raw_segments with page ranges.
        for head, body_lines in groups:
            body_text = " ".join(ln for (_, ln) in body_lines)
            text = (head + "\n" + body_text).strip() if body_text.strip() else head.strip()
            if len(text) < 20:
                continue
            pstart = body_lines[0][0] if body_lines else (lines[0][0])
            # char offsets: locate head/body in stream best-effort
            idx_h = stream.find(head) if head in stream else -1
            cstart = idx_h if idx_h >= 0 else 0
            raw_segments.append((head, body_text, pstart, cstart))
    else:
        # numbered inline markers
        cuts = [(m.start(), m.group(0).strip(), m.end()) for m in SECTION_RE.finditer(stream)]
        if not cuts:
            raw_segments.append((path.stem.replace("_", " "), stream, 1, 0))
        else:
            if cuts[0][0] > 0:
                raw_segments.append((path.stem.replace("_", " "), stream[: cuts[0][0]],
                                     _page_at(offsets, 0), 0))
            for i, (start, head, end) in enumerate(cuts):
                stop = cuts[i + 1][0] if i + 1 < len(cuts) else len(stream)
                raw_segments.append((head, stream[end:stop].strip(),
                                     _page_at(offsets, start), start))

    sections: list[Section] = []
    # Track which pages are covered by a detected section.
    covered_pages: set[int] = set()

    for idx, (head, body, pstart, cstart) in enumerate(raw_segments):
        if not body.strip() and not head.strip():
            continue
        # page_end: page at end of body
        cend = cstart + len(head) + 1 + len(body)
        pend = _page_at(offsets, min(cend - 1, len(stream) - 1))
        text = (head + "\n" + body).strip() if body.strip() else head.strip()
        # Split body into clauses.
        clauses = _split_clauses(body, pstart, offsets, cstart + len(head) + 1)
        sec = Section(
            section_id="",  # assigned by caller/indexer
            heading=head,
            level=1,
            page_start=pstart,
            page_end=max(pstart, pend),
            text=text,
            char_start=cstart,
            char_end=cend,
            clauses=clauses,
        )
        sections.append(sec)
        for p in range(sec.page_start, sec.page_end + 1):
            covered_pages.add(p)

    # Synthetic sections for orphan pages (no heading detected at all, or gaps).
    if not sections:
        # Whole doc as one section.
        sections.append(Section(
            section_id="",
            heading=path.stem.replace("_", " "),
            level=1,
            page_start=1,
            page_end=pages,
            text=stream,
            char_start=0,
            char_end=len(stream),
            clauses=_split_clauses(stream, 1, offsets, 0),
        ))
        covered_pages = set(range(1, pages + 1))

    # Add synthetic page sections for any page not covered.
    for p in range(1, pages + 1):
        if p in covered_pages:
            continue
        # Find text for that page: lines on page p.
        page_text = " ".join(line for (pn, line) in lines if pn == p)
        if not page_text.strip():
            continue
        sections.append(Section(
            section_id="",
            heading=f"Page {p}",
            level=1,
            page_start=p,
            page_end=p,
            text=page_text,
            char_start=0,
            char_end=len(page_text),
            clauses=[],
        ))

    # Assign ordinals/ids later; set temporary ids here using index.
    for i, s in enumerate(sections):
        s.section_id = f"s{i}"

    return DocumentTree(sha256=sha256, filename=path.name, pages=pages, sections=sections)


def _split_clauses(body: str, page_start: int, offsets, base_offset: int) -> list[Clause]:
    """Split a section body into numbered clauses."""
    clauses: list[Clause] = []
    # Find clause markers.
    matches = list(CLAUSE_RE.finditer(body))
    if not matches:
        # Whole body as one pseudo-clause if non-trivial.
        if body.strip():
            clauses.append(Clause(clause_id="", text=body.strip(),
                                 page=page_start, char_start=base_offset,
                                 char_end=base_offset + len(body)))
        return clauses
    for i, m in enumerate(matches):
        start = m.start()
        cid = m.group(1)
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        text = body[start:end].strip()
        if not text:
            continue
        clauses.append(Clause(
            clause_id=cid,
            text=text,
            page=page_start,
            char_start=base_offset + start,
            char_end=base_offset + end,
        ))
    # Leading text before first clause marker.
    if matches and matches[0].start() > 0:
        lead = body[: matches[0].start()].strip()
        if lead:
            clauses.insert(0, Clause(clause_id="", text=lead, page=page_start,
                                    char_start=base_offset,
                                    char_end=base_offset + matches[0].start()))
    return clauses


def assign_ids(tree: DocumentTree) -> None:
    """Assign stable section ids based on doc sha. Call after sha known."""
    for i, s in enumerate(tree.sections):
        s.section_id = f"{tree.sha256[:8]}:s{i}"
        for j, c in enumerate(s.clauses):
            if not c.clause_id:
                c.clause_id = f"{s.section_id}.c{j}"


def build_trees(paths: Iterable[Path]) -> list[DocumentTree]:
    out = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            for pdf in sorted(p.rglob("*.pdf")):
                out.append(build_tree(pdf))
        else:
            out.append(build_tree(p))
    for t in out:
        assign_ids(t)
    return out


# ---------------------------------------------------------------------------
# Bridge to core.index.ExtractedDoc (sections become the retrieval units)
# ---------------------------------------------------------------------------

def _count_tokens(text: str, tokenizer=None) -> int:
    if tokenizer is not None:
        try:
            return len(tokenizer.encode(text).ids)
        except Exception:
            pass
    return int(len(text) / 3.5) + 1


def to_extracted_doc(tree: DocumentTree, budget: int = 400, tokenizer=None) -> "ExtractedDoc":
    """Convert a DocumentTree into an ExtractedDoc whose chunks are sections.

    Each section becomes one retrieval unit (its full text, unchunked). The
    400-token budget is NOT asserted here: sections may exceed it; the encoder
    truncates for embedding and generation applies its own context budget.
    """
    import sys as _sys
    from pathlib import Path as _P
    _src = _P(__file__).resolve().parents[1]
    if str(_src) not in _sys.path:
        _sys.path.insert(0, str(_src))
    from core.index import ExtractedDoc, ExtractedChunk  # noqa: E402

    chunks = []
    for s in tree.sections:
        chunks.append(ExtractedChunk(
            page=s.page_start,
            char_start=s.char_start,
            char_end=s.char_end,
            text=s.text,
            n_tokens=_count_tokens(s.text, tokenizer),
        ))
    if not chunks:
        # Fallback: whole document as one section.
        chunks.append(ExtractedChunk(page=1, char_start=0, char_end=0,
                                     text="", n_tokens=0))
    return ExtractedDoc(sha256=tree.sha256, filename=tree.filename,
                        pages=tree.pages, chunks=chunks)


def section_metadata(tree: DocumentTree) -> list[dict]:
    """Sidecar metadata per section, keyed by ordinal (matches ExtractedDoc chunks)."""
    out = []
    for i, s in enumerate(tree.sections):
        out.append({
            "ordinal": i,
            "section_id": s.section_id,
            "heading": s.heading,
            "level": s.level,
            "page_start": s.page_start,
            "page_end": s.page_end,
            "char_start": s.char_start,
            "char_end": s.char_end,
            "n_clauses": len(s.clauses),
            "summary": s.summary or _first_sentences(s.text, 2),
        })
    return out