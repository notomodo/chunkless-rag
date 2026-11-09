#!/usr/bin/env python3
"""UI-agnostic glue for the application layer — the first place the whole
pipeline (extract → chunk → embed → persist → retrieve → generate) runs end to
end on real components instead of an eval harness or a stub.

WHY THIS FILE PRINTS NOTHING
============================
Presentation is `cli.py`'s job. This module returns STRUCTURED results
(`IngestSummary`, `AnswerResult`) and reports progress through optional
callbacks. That split is what lets the same glue be driven by the CLI today and
tested headless (no stdout scraping) in `tests/test_cli.py`.

WHY THE ENCODER IS BUILT ONCE, AND INJECTABLE
=============================================
`Index.append_document` and `Index.search` both take an `IdentifiedEncoder`,
and the index's manifest FATALLY rejects a search whose `embedder_id` /
`tokenizer_sha256` differ from what it was built with (mixing embedding spaces
makes every dense score meaningless — see core/index.py). So ingest and ask
MUST use the same encoder identity; this module owns that single handle and
reuses it for both. The handle is also constructor-injectable so tests can pass
a deterministic `FakeEncoder` and exercise every offline code path (idempotency,
the empty-index guard, the embedder-mismatch guard) without onnxruntime, a real
.onnx file, or Ollama.

REUSE, NOT REINVENTION
======================
  * ExtractedDoc bridging (incl. faithful char offsets) — `scripts/
    migrate_chunk_ids.py._extract_doc`, imported and called verbatim. It
    reconstructs offsets by locating each chunk body in the per-document line
    stream and RAISES if a body can't be found, so an offset is faithful or
    ingestion aborts — it never writes a guessed or zero span silently.
  * The grounding prompt — `gen_answer.SYSTEM_PROMPT` / `USER_TEMPLATE`,
    imported unchanged (DECISION-004's v3 policy: answer only from context,
    else the bare sentinel NOT_IN_DOCUMENTS).
  * Retrieval + persistence — `core.index.Index`, untouched.

Only generation is new here: `gen_answer.call_ollama` hard-codes stream:false,
which at 65–105 s/question is indistinguishable from a hang. We add a STREAMING
variant with byte-identical options (temperature 0, seed 42, num_ctx 4096) over
stdlib urllib — the exact transport gen_answer already uses, no new dependency.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

# This package's modules live under src/ alongside the flat top-level modules
# (retriever, ingest_sme, gen_answer) and the core package; the ExtractedDoc
# bridge lives under scripts/. Mirror the sys.path pattern the rest of the repo
# already uses so this file imports cleanly whether it's loaded as `app.pipeline`
# (installed / from src on the path) or stand-alone (the import smoke test).
_SRC = Path(__file__).resolve().parents[1]
_ROOT = _SRC.parent
for _p in (str(_SRC), str(_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from core.index import (  # noqa: E402
    EncoderHandle,
    ExtractedDoc,
    Hit,
    Index,
    IndexStats,
)
from gen_answer import SYSTEM_PROMPT, USER_TEMPLATE  # noqa: E402 -- v3 grounding prompt, verbatim

# The exact sentinel the v3 prompt tells the model to emit when the context
# contains nothing that answers the question. Detecting it is how abstention
# "shows its work" (near-misses) instead of being a black box.
ABSTAIN_SENTINEL = "NOT_IN_DOCUMENTS"

MODEL_NAME = "BAAI/bge-small-en-v1.5"
DEFAULT_ONNX_PATH = _ROOT / "models" / "bge-small-en-v1.5.onnx"
DEFAULT_TOKENIZER_PATH = _SRC / "tokenizer.json"
DEFAULT_OLLAMA_MODEL = "qwen2.5:3b-instruct"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_BUDGET = 400
DEFAULT_K = 3


class EmptyIndexError(RuntimeError):
    """Raised by `answer()` when the index holds no documents. A pre-generation
    guard: it fires before the encoder or Ollama are touched, so the CLI can
    turn it into actionable guidance ("run: adtc-rag ingest <path>") and a
    non-zero exit instead of an empty, confusing answer."""


class ModelNotFoundError(RuntimeError):
    """Raised when the shipping .onnx encoder file is absent — ingest and ask
    both need real vectors, and there is no meaningful fallback."""


# =============================================================================
# Structured results (no printing — cli.py formats these)
# =============================================================================


@dataclass
class DocIngestResult:
    filename: str
    sha256: str
    already_indexed: bool
    n_chunks: int
    pages: int


@dataclass
class IngestSummary:
    docs: list[DocIngestResult]
    stats: IndexStats
    encoder_init_s: float
    total_s: float

    @property
    def n_newly_indexed(self) -> int:
        return sum(1 for d in self.docs if not d.already_indexed)


@dataclass
class AnswerResult:
    question: str
    answer: str
    abstained: bool
    hits: list[Hit]            # the k passages actually put in context
    considered: list[Hit]      # near-misses just below the cut (the "why")
    timings: dict[str, float]  # SearchResult.timings (bm25_ms/dense_ms/fuse_ms/total_ms)
    encoder_init_s: float
    retrieval_s: float
    generation_s: float
    # Ollama's own generation metrics from the terminal "done" object:
    # prompt_eval_count / prompt_eval_duration (ns) / eval_count / eval_duration (ns)
    # (+ total_duration, load_duration when present). None if the stream ended
    # without a done object. These are the model's self-timed counters — the
    # basis for tokens/sec, which wall-clock (encoder init, HTTP, streaming
    # overhead) would distort. See _stream_generate.
    gen_stats: dict | None = None


# Progress callbacks. Ingest emits ("extracting", name) / ("embedding", name, n)
# / ("appended", name, DocIngestResult) / ("already_indexed", name, DocIngestResult).
# Answer emits a single phase string ("retrieving" | "generating") and streams
# generated tokens one at a time.
IngestEvent = tuple
OnIngestEvent = Callable[[IngestEvent], None]
OnPhase = Callable[[str], None]
OnToken = Callable[[str], None]


# =============================================================================
# Pipeline
# =============================================================================


class Pipeline:
    """Owns the single encoder identity shared by ingest and ask, plus the
    index location and generation settings. Construct once per process."""

    def __init__(
        self,
        index_dir: Path | None = None,
        encoder=None,
        onnx_path: Path | None = None,
        tokenizer_path: Path | None = None,
        budget: int = DEFAULT_BUDGET,
        model: str = DEFAULT_OLLAMA_MODEL,
        host: str = DEFAULT_OLLAMA_HOST,
        think: bool | None = None,
        chunkless: bool = False,
        selector: str = "batch",
        use_navigator: bool = True,
    ) -> None:
        self.index_dir = Path(index_dir).expanduser() if index_dir is not None else None
        self.onnx_path = Path(onnx_path) if onnx_path is not None else DEFAULT_ONNX_PATH
        self.tokenizer_path = Path(tokenizer_path) if tokenizer_path is not None else DEFAULT_TOKENIZER_PATH
        self.budget = budget
        self.model = model
        self.host = host
        self.chunkless = chunkless
        self.selector = selector
        self.use_navigator = use_navigator
        # Thinking-mode control for hybrid models (Qwen3): None => omit the field
        # entirely, so the payload is byte-identical to what the non-thinking
        # Qwen2.5 models were benchmarked with. Set False to disable a hybrid
        # model's reasoning trace (keeps it in the same "answer directly from
        # context" regime as the 2.5 instruct models); True to enable it.
        self.think = think

        # An injected encoder (tests) short-circuits the real ONNX build and its
        # init cost is reported as zero. The real handle is built lazily on first
        # use so the empty-index guard and --help never pay for onnxruntime.
        self._encoder = encoder
        self._encoder_init_s = 0.0
        self._tok = None

    # -- lazily built, cached components ----------------------------------

    def _get_encoder(self):
        """Build the shipping OnnxEncoder + EncoderHandle once, timing the init.
        Identity strings match scripts/migrate_chunk_ids.py exactly so an index
        built by either path is interchangeable."""
        if self._encoder is not None:
            return self._encoder
        if not self.onnx_path.exists():
            raise ModelNotFoundError(
                f"ONNX encoder not found at {self.onnx_path}. It is gitignored and "
                f"built from source — run:  python scripts/export_onnx.py"
            )
        if not self.tokenizer_path.exists():
            raise ModelNotFoundError(f"vendored tokenizer not found at {self.tokenizer_path}")

        from retriever import OnnxEncoder  # local: onnxruntime is runtime-only, not a test dep

        t0 = time.perf_counter()
        tok_sha = hashlib.sha256(self.tokenizer_path.read_bytes()).hexdigest()
        enc = OnnxEncoder(
            str(self.onnx_path),
            MODEL_NAME,
            model_name=MODEL_NAME,
            tokenizer_path=self.tokenizer_path,
        )
        self._encoder = EncoderHandle(
            encoder=enc, embedder_id=MODEL_NAME, tokenizer_sha256=tok_sha
        )
        self._encoder_init_s = time.perf_counter() - t0
        return self._encoder

    def _get_tokenizer(self):
        """The tokenizers.Tokenizer used for budget-accurate chunking during
        extraction. Vendored + offline, same file the encoder identity hashes."""
        if self._tok is None:
            from tokenizers import Tokenizer  # local: optional dep

            if not self.tokenizer_path.exists():
                raise ModelNotFoundError(f"vendored tokenizer not found at {self.tokenizer_path}")
            self._tok = Tokenizer.from_file(str(self.tokenizer_path))
        return self._tok

    def _open_index(self) -> Index:
        return Index.open(self.index_dir)

    # -- ingest ------------------------------------------------------------

    def ingest_path(self, paths: Sequence[str | Path], on_event: OnIngestEvent | None = None) -> IngestSummary:
        """Ingest PDF paths and/or directories (recursed for *.pdf) into the
        index. Idempotent per-document by sha256: an already-indexed file is
        skipped WITHOUT re-extracting (extraction is the slow part), reported as
        already-indexed. Emits progress events so a multi-minute embed is never
        a silent hang."""
        t_start = time.perf_counter()
        pdfs = self._collect_pdfs(paths)
        idx = self._open_index()
        encoder = self._get_encoder()  # build now so its cost is attributed to init, not doc 1

        results: list[DocIngestResult] = []
        for pdf in pdfs:
            sha = hashlib.sha256(pdf.read_bytes()).hexdigest()
            if idx.has_document(sha):
                info = idx.manifest["documents"][sha]
                r = DocIngestResult(pdf.name, sha, True, info["n_chunks"], info["pages"])
                results.append(r)
                _emit(on_event, ("already_indexed", pdf.name, r))
                continue

            _emit(on_event, ("extracting", pdf.name))
            doc = self._extract(pdf)
            results.extend(self._ingest_docs(idx, [doc], encoder, on_event))

        return IngestSummary(
            docs=results,
            stats=idx.stats(),
            encoder_init_s=self._encoder_init_s,
            total_s=time.perf_counter() - t_start,
        )

    def _ingest_docs(
        self, idx: Index, docs: Iterable[ExtractedDoc], encoder, on_event: OnIngestEvent | None
    ) -> list[DocIngestResult]:
        """Append pre-extracted ExtractedDocs. Split out from ingest_path so the
        append/idempotency/summary logic is testable with hand-built docs and a
        FakeEncoder — no PDF, no tokenizer, no onnxruntime."""
        out: list[DocIngestResult] = []
        for doc in docs:
            if idx.has_document(doc.sha256):
                info = idx.manifest["documents"][doc.sha256]
                r = DocIngestResult(doc.filename, doc.sha256, True, info["n_chunks"], info["pages"])
                out.append(r)
                _emit(on_event, ("already_indexed", doc.filename, r))
                continue
            _emit(on_event, ("embedding", doc.filename, len(doc.chunks)))
            res = idx.append_document(doc, encoder)
            r = DocIngestResult(doc.filename, doc.sha256, False, res.n_chunks_added, doc.pages)
            out.append(r)
            _emit(on_event, ("appended", doc.filename, r))
        return out

    def _extract(self, pdf: Path) -> ExtractedDoc:
        if not self.chunkless:
            import migrate_chunk_ids  # scripts/: reuse _extract_doc, do not reinvent

            return migrate_chunk_ids._extract_doc(pdf, self.budget, self._get_tokenizer())

        # Chunkless path: structure extraction -> sections as retrieval units.
        import hashlib
        import json
        import sys
        from pathlib import Path as _Path

        _src = _Path(__file__).resolve().parents[1]
        _root = _src.parent
        for _p in (str(_src), str(_root / "scripts")):
            if _p not in sys.path:
                sys.path.insert(0, _p)

        import structure  # noqa: E402
        import summarise  # noqa: E402

        tree = structure.build_tree(pdf)
        structure.assign_ids(tree)
        tok = self._get_tokenizer()
        # summaries (extractive by default; LLM optional via env not wired here)
        metas = structure.section_metadata(tree)
        full_texts = [s.text for s in tree.sections]
        summaries = summarise.summarise_sections(metas, full_texts, use_llm=False)
        for m, s, sec in zip(metas, summaries, tree.sections):
            m["summary"] = s
            sec.summary = s

        # sidecar: persist section metadata (heading, page range, summary) per doc
        self._write_sections_meta(tree.sha256, metas)

        return structure.to_extracted_doc(tree, self.budget, tok)

    def _write_sections_meta(self, doc_sha: str, metas: list) -> None:
        if self.index_dir is None:
            return
        p = Path(self.index_dir).expanduser() / "sections_meta.json"
        data = {}
        if p.exists():
            try:
                data = json.loads(p.read_text())
            except Exception:
                data = {}
        data[doc_sha] = metas
        p.write_text(json.dumps(data, indent=2))

    def _load_sections_meta(self) -> dict:
        if self.index_dir is None:
            return {}
        p = Path(self.index_dir).expanduser() / "sections_meta.json"
        if not p.exists():
            return {}
        try:
            return json.loads(p.read_text())
        except Exception:
            return {}

    def _chunkless_navigate(self, idx, question: str, k: int):
        import sys
        from pathlib import Path as _Path
        _src = _Path(__file__).resolve().parents[1]
        if str(_src) not in sys.path:
            sys.path.insert(0, str(_src))
        import navigator as _nav  # noqa: E402
        from core.index import Hit, SearchResult  # noqa: E402

        records = idx._read_all_chunks()
        metas_all = self._load_sections_meta()
        # map full sha by prefix from manifest
        sha_by_prefix = {s[:8]: s for s in idx.manifest.get("documents", {}).keys()}

        outline = []
        for i, r in enumerate(records):
            full_sha = sha_by_prefix.get(r["doc_sha256"][:8], r["doc_sha256"])
            doc_metas = metas_all.get(full_sha, [])
            ordinal = int(r["id"].split(":")[1]) if ":" in r["id"] else i
            m = doc_metas[ordinal] if ordinal < len(doc_metas) else {}
            outline.append({
                "idx": i,
                "heading": m.get("heading") or r["text"].split("\n", 1)[0][:80],
                "pages": f"{r['page']}-{m.get('page_end', r['page'])}",
                "summary": m.get("summary") or r["text"][:200],
            })

        chosen = _nav.navigate(outline, question, strategy=self.selector, k=k,
                               model=self.model, host=self.host)
        if not chosen:
            return None
        recs = [records[i] for i in chosen if 0 <= i < len(records)]
        if not recs:
            return None
        hits = []
        for r in recs:
            hits.append(Hit(
                id=r["id"], text=r["text"], filename=r["filename"], page=r["page"],
                char_start=r["char_start"], char_end=r["char_end"],
                rrf_score=0.0, bm25_rank=None, dense_rank=None,
            ))
        return SearchResult(hits=hits, considered=[], timings={
            "bm25_ms": 0.0, "dense_ms": 0.0, "fuse_ms": 0.0, "total_ms": 0.0,
        })

    @staticmethod
    def _collect_pdfs(paths: Sequence[str | Path]) -> list[Path]:
        found: list[Path] = []
        seen: set[Path] = set()
        for raw in paths:
            p = Path(raw).expanduser()
            candidates = sorted(p.rglob("*.pdf")) if p.is_dir() else [p]
            for c in candidates:
                rp = c.resolve()
                if rp in seen:
                    continue
                if not c.exists():
                    raise FileNotFoundError(f"no such path: {c}")
                if c.suffix.lower() != ".pdf":
                    raise ValueError(f"not a PDF: {c}")
                seen.add(rp)
                found.append(c)
        if not found:
            raise FileNotFoundError(f"no PDFs found in: {', '.join(str(p) for p in paths)}")
        return found

    # -- ask ---------------------------------------------------------------

    def answer(
        self,
        question: str,
        k: int = DEFAULT_K,
        on_phase: OnPhase | None = None,
        on_token: OnToken | None = None,
    ) -> AnswerResult:
        """Retrieve k passages and generate a grounded, streamed answer.

        Order matters for offline-testability: the empty-index guard fires
        BEFORE the encoder or Ollama are touched, so an empty index is a clean
        non-zero exit rather than a crash, and that guard is testable without
        any heavy dependency.
        """
        idx = self._open_index()
        if idx.stats().n_documents == 0:
            raise EmptyIndexError("no documents indexed")

        encoder = self._get_encoder()

        if on_phase:
            on_phase("retrieving")
        t0 = time.perf_counter()
        sr = None
        if self.chunkless and self.use_navigator:
            try:
                sr = self._chunkless_navigate(idx, question, k)
            except Exception:
                sr = None
        if sr is None:
            sr = idx.search(question, k=k, encoder=encoder)
        retrieval_s = time.perf_counter() - t0

        if on_phase:
            on_phase("generating")
        context = "\n\n".join(f"[{i}] {h.text}" for i, h in enumerate(sr.hits, 1))
        user = USER_TEMPLATE.format(context=context, question=question)
        t1 = time.perf_counter()
        text, gen_stats = self._stream_generate(user, on_token)
        generation_s = time.perf_counter() - t1

        return AnswerResult(
            question=question,
            answer=text,
            abstained=text.strip().startswith(ABSTAIN_SENTINEL),
            hits=sr.hits,
            considered=sr.considered,
            timings=sr.timings,
            encoder_init_s=self._encoder_init_s,
            retrieval_s=retrieval_s,
            generation_s=generation_s,
            gen_stats=gen_stats,
        )

    def _build_payload(self, user: str) -> dict:
        """The exact /api/chat body. Options are byte-identical to
        gen_answer.call_ollama (temperature 0, seed 42, num_ctx 4096); only
        stream:true differs. `think` is added ONLY when explicitly set, so a
        Pipeline constructed without it (every Qwen2.5 run) produces a payload
        indistinguishable from the originally benchmarked one."""
        payload: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            "stream": True,
            "options": {"temperature": 0, "seed": 42, "num_ctx": 4096},
        }
        if self.think is not None:
            payload["think"] = self.think
        return payload

    def _stream_generate(self, user: str, on_token: OnToken | None) -> tuple[str, dict | None]:
        """Stream tokens from local Ollama. Options are byte-identical to
        gen_answer.call_ollama (temperature 0, seed 42, num_ctx 4096) so the CLI
        reproduces the benchmarked generation exactly — only stream:true differs.
        Stdlib urllib, the same transport gen_answer already uses.

        Returns (text, gen_stats). gen_stats is the model's self-reported metrics
        harvested from the terminal "done" object — prompt_eval_count /
        prompt_eval_duration / eval_count / eval_duration (+ total/load_duration).
        The benchmark harness turns these into tokens/sec; wall-clock can't,
        because it folds in HTTP and streaming overhead. Metrics only — the
        payload is unchanged, so this streams byte-identically to before."""
        import urllib.request

        payload = self._build_payload(user)
        req = urllib.request.Request(
            f"{self.host}/api/chat",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        parts: list[str] = []
        gen_stats: dict | None = None
        # Ollama streams newline-delimited JSON objects, one per token-ish chunk,
        # terminated by an object with "done": true — which also carries the
        # generation counters. We keep that final object's metric fields.
        _METRIC_KEYS = (
            "total_duration", "load_duration",
            "prompt_eval_count", "prompt_eval_duration",
            "eval_count", "eval_duration",
        )
        with urllib.request.urlopen(req, timeout=600) as resp:
            for raw in resp:
                raw = raw.strip()
                if not raw:
                    continue
                obj = json.loads(raw)
                tok = obj.get("message", {}).get("content", "")
                if tok:
                    parts.append(tok)
                    if on_token:
                        on_token(tok)
                if obj.get("done"):
                    gen_stats = {k: obj[k] for k in _METRIC_KEYS if k in obj}
                    break
        return "".join(parts), gen_stats


def _emit(cb: OnIngestEvent | None, event: IngestEvent) -> None:
    if cb is not None:
        cb(event)
