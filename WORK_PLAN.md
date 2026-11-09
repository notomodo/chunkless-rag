# Work Plan — Chunkless RAG for SMEs

Adapt `adtc-2026` (offline, CPU-only RAG over SME policy/contract PDFs) into a
**chunkless** variant: retrieval navigates document structure (sections/pages)
instead of fixed-size chunks, per the DeepRead / Adaptive-Query-Routing (PageIndex,
tree reasoning) approach. The papers that ground this plan are in `papers/`.

## 0. Baseline we are changing (adtc-2026)

- Extract: `src/ingestion/extract.py` (table-aware) + `src/ingest_sme.py`
  (prose-first, structure-aware **chunking** → 47 chunks, 400-token budget asserted).
- Retrieve: `src/retriever.py` — hand-rolled BM25 + pluggable dense encoder, RRF
  (k=60); pure-BM25 fallback.
- Generate: `src/gen_answer.py` — Qwen2.5-3B via local Ollama, locked v3 prompt,
  top-k=3 chunks, abstention sentinel `NOT_IN_DOCUMENTS`.
- Persist: `src/core/index.py` — append-only index keyed by chunk rows,
  embedder/tokenizer identity guards, atomic writes, crash self-healing.
- App: `src/app/pipeline.py` + `src/app/cli.py` — `ingest` / `ask`, offline.

The chunk-based pipeline is the **control**. This plan produces a parallel
**chunkless** pipeline that reuses the locked pieces (BM25 formula, RRF, v3
prompt, ONNX encoder, Ollama transport, index atomicity) but changes the
retrieval unit from "chunk" to "section/page".

## 1. Target architecture (chunkless)

```
PDF ──▶ structure extraction ──▶ section/page tree ──▶ summarisation ──▶ reasoning retrieval ──▶ LLM answer
        (pdfplumber/Docling)    (headings, pages,      (per-section      (LLM navigates outline,    (Qwen2.5-3B,
                                 clause ranges)         + doc summary)    BM25 candidate pre-filter)   v3 prompt)
```

Key differences from baseline:
- **No fixed 400-token chunking.** The retrieval unit is a *section* (heading +
  its body, possibly spanning multiple pages) or a *page* when no heading exists.
- **Two-stage retrieval.** (a) cheap lexical pre-filter (BM25 over section text
  and/or summaries) to get a candidate set; (b) LLM reasoning over the document
  outline + candidate summaries to select the sections/pages to read (DeepRead
  "locate-then-read"; TreeGuidedPageSelector-style traversal).
- **Long-context aware.** Sections can exceed the model's context; generation
  must truncate/summarise or split at sentence boundaries, respecting "lost in
  the middle" (position-aware ordering of selected sections).
- **Citations** become `[file, section, p.range]` instead of `[file, p, chunk_id]`.

## 2. Phases and milestones

### Phase 0 — Repo & scaffold (0.5 day)
- [ ] Create GitHub repo `chunkless-rag` (public, MIT), push this baseline + plan.
- [ ] Rename package identity where it leaks `adtc` (CLI prog name, index dir
  default `~/.chunkless/index`) — keep `adtc` strings only in historical docs.
- [ ] Add `papers/` (already copied) and `WORK_PLAN.md`.

### Phase 1 — Structure extraction (2 days)
- [ ] New module `src/ingestion/structure.py`: PDF → `DocumentTree` of
  `Section(id, heading, level, page_start, page_end, text, parent_id)`.
  - Reuse `extract_lines` page-furniture removal and section-marker regex from
    `ingest_sme.py`, but **keep** hierarchy instead of flattening into chunks.
  - Fallback: pages with no detectable heading become synthetic sections
    (`Section (page N)`), so every page is reachable.
- [ ] Preserve char offsets per section for faithful citations (reuse the
  `_extract_doc` bridge pattern in `scripts/migrate_chunk_ids.py`).
- [ ] Tests: `tests/test_structure.py` — golden tree for the Kibuga corpus
  (expected section count, page coverage, no orphan pages).

### Phase 2 — Chunkless index (2 days)
- [ ] Generalise `src/core/index.py`'s `ExtractedChunk`/`ExtractedDoc` into
  `Section`/`SectionedDoc` (or add `unit="section"` metadata) without breaking
  the chunk-based control. Decision: keep a separate `core/section_index.py` to
  avoid destabilising the baseline index schema; revisit merge later.
- [ ] Persist: `sections.jsonl` (id, doc_sha, heading, level, page range,
  char span, text, summary), `section_embeddings.npy` (optional, over summaries),
  `bm25_sections.json`, `manifest.json` — same atomic-write + crash-heal
  contract as `index.py`.
- [ ] Stable ids `f"{doc_sha[:8]}:s{ordinal}"`, append-only, idempotent by
  doc sha.

### Phase 3 — Summarisation (1 day)
- [ ] `src/summarise.py`: per-section summary (1–3 sentences) + document-level
  summary (first 3 pages), via local Ollama (Qwen2.5-3B) OR a deterministic
  extractive fallback for fully-offline no-LLM runs. Cache by (section text
  hash) so re-ingest is free.
- [ ] Granularity flag: `--summary-level element|page` (mirrors docling-agent).

### Phase 4 — Retrieval (3 days)
- [ ] `src/retrieval/section_bm25.py`: BM25 over section text + summaries
  (reuse `retriever.BM25`, no reimplemented formula).
- [ ] `src/retrieval/navigator.py`: LLM reasoning selector:
      input = (query, document outline with section summaries); output = ordered
      list of section ids to read. Implement two selectors:
      - `BatchSelector` (flat, evaluate candidates in batches, early stopping)
      - `TreeSelector` (top-down traversal, follow cross-refs like "See §4")
  - Use Ollama structured output (JSON) with the same retry/abstain discipline
    as gen_answer.
- [ ] `src/retrieval/chunkless_retriever.py`: orchestrate BM25 pre-filter →
  navigator → read selected sections → position-aware ordering (mitigate
  lost-in-the-middle: put highest-relevance section first/last, not middle).
- [ ] Keep pure-BM25 fallback (`navigator=None`) for offline/no-LLM mode and
  for honest ablation vs. the chunk-based baseline.

### Phase 5 — Generation (1 day)
- [ ] Reuse `gen_answer.SYSTEM_PROMPT`/`USER_TEMPLATE` (v3 grounding policy).
- [ ] Build context from selected sections with headings + page ranges; enforce a
  context budget (truncate least-relevant sections first) instead of the 400-token
  chunk budget.
- [ ] Abstention sentinel + near-miss reporting adapted to sections.

### Phase 6 — Evaluation & reproducibility (2 days)
- [ ] Re-label gold set: question → section_id(s) instead of chunk_id; provide a
  migration script from the existing 19/35-q sets where possible (many gold
  chunks map to a parent section).
- [ ] Retrieval metrics (R@1/R@3/R@5/MRR) at section level; generation metrics
  (Layer A checker) unchanged.
- [ ] Reproducibility gate: section-tree fingerprint (hash of section headings +
  page ranges + text), analogous to the 47-chunk fingerprint.
- [ ] Head-to-head benchmark: chunk-based (control) vs chunkless (new) on the
  same 35 questions, same LLM/prompt, report delta. This is the evidence the
  adaptation is worth it.

### Phase 7 — Offline packaging & tests (1.5 days)
- [ ] ONNX encoder over **section summaries** (or reuse existing encoder on
  section text); vendored tokenizer unchanged.
- [ ] CLI: `chunkless-rag ingest <pdfs>` / `ask "<q>"` with `--selector
  batch|tree`, `--no-navigator` (BM25-only) flags.
- [ ] Tests: structure extraction, section index append/idempotency/crash-heal,
  navigator JSON parsing, retrieval metrics, end-to-end offline smoke test.

### Phase 8 — Docs & submission pack (1 day)
- [ ] DECISION docs: `DECISION-006-chunkless-unit.md`,
  `DECISION-007-navigator.md`, `DECISION-008-context-budget.md`.
- [ ] Update README architecture diagram and quickstart.
- [ ] Results table in `REPORT.md` (chunk vs chunkless).

## 3. Reuse vs. rewrite matrix

| Component | Reuse | Rewrite |
|---|---|---|
| BM25 formula (`retriever.BM25`) | ✅ verbatim | — |
| RRF (`retriever.rrf_fuse`) | ✅ verbatim | — |
| v3 grounding prompt | ✅ verbatim | — |
| ONNX encoder + vendored tokenizer | ✅ | — |
| Ollama streaming transport | ✅ | — |
| Index atomicity/crash-heal pattern | ✅ pattern | section schema |
| Chunker (`ingest_sme.py`) | ❌ | → `structure.py` |
| HybridRetriever orchestration | ❌ | → section retriever + navigator |
| Gold labels (chunk ids) | migrate | → section ids |
| Corpus fingerprint | adapt | section-tree fingerprint |

## 4. Risks and mitigations

1. **Lost-in-the-middle on long sections.** Mitigation: section-level retrieval +
   position-aware ordering + context budget; measure with the 35-q set.
2. **Navigator hallucinating section ids.** Mitigation: validate returned ids
   against the outline; fall back to BM25 top-k on parse failure; log.
3. **LLM cost/latency on CPU.** Two LLM calls per query (navigate + answer).
   Mitigation: BM25 pre-filter caps candidate outline size; cache summaries;
   `--no-navigator` mode for low-latency.
4. **Reproducibility drift.** Mitigation: section-tree fingerprint gate, vendored
   tokenizer, pinned deps (carry over).
5. **Evaluation comparability.** Section-level gold labels are coarser; report
   both chunk-level (control) and section-level (new) numbers, clearly labelled.

## 5. Definition of done

- `chunkless-rag ingest <pdfs>` builds a section index offline.
- `chunkless-rag ask "<q>"` streams a grounded answer citing sections/pages.
- Head-to-head report shows chunkless ≥ chunk baseline on Recall@5/MRR or
  documents the trade-off honestly.
- All tests pass offline (no network, no torch on shipping path).
- New GitHub repo published with MIT licence and reproducible quickstart.

## 6. Immediate next steps

1. Create GitHub repo and push baseline (Phase 0).
2. Implement `src/ingestion/structure.py` + tests (Phase 1) before touching
   retrieval, so the new unit is locked and reviewable.
3. Keep the chunk-based pipeline runnable as the control throughout.
