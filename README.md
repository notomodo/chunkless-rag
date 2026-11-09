# adtc-2026

**An offline, CPU-only RAG document-Q&A assistant for SMEs.**
Built for the Africa Deep Tech Challenge 2026.

Small and medium enterprises hold their rules in PDFs — terms, policies, contracts —
and need plain-language answers grounded in those documents. This system retrieves the
relevant passages and has a small local LLM answer from them. It runs **entirely
offline on an 8 GB CPU-only machine**: no API calls, no network at runtime. Offline
operation and reproducibility from a clean clone are judged criteria, so they are
enforced in code, not left to convention.

## Architecture

```
   PDF ──▶ extraction ──▶ chunking ──▶ hybrid retrieval ──▶ LLM answer
           (pdfplumber)   (structure-   (BM25 + dense,        (Qwen2.5-3B,
                           aware)         RRF fusion)           grounded, v3 prompt)
```

- **Extraction** — `src/ingestion/extract.py`: table-aware extractor for the financial
  corpus (headers carried into serialised rows). See `docs/REPO_CONSOLIDATION.md`.
- **Chunking** — `src/ingest_sme.py`: prose-first, structure-aware. Splits on clause and
  section boundaries (not blind character counts) and carries the section heading into
  every chunk. Token budget is **asserted** against the vendored tokenizer, never
  estimated.
- **Retrieval** — `src/retriever.py`: hand-rolled Okapi **BM25** (stdlib only) fused with
  a pluggable **dense encoder** via **Reciprocal Rank Fusion** (RRF, k=60). Pure-BM25
  fallback with an identical API (`encoder=None`).
- **LLM answer layer** — `src/gen_answer.py`: Qwen2.5-3B-Instruct via local Ollama, grounded
  on the top-`k=3` retrieved chunks under the locked v3 prompt. See
  [`docs/DECISION-004-grounding-prompt.md`](docs/DECISION-004-grounding-prompt.md).

## Retrieval result

Nine configurations benchmarked against a **47-chunk corpus of real SME documents**
(Kibuga, a Ugandan e-commerce marketplace — terms, privacy, returns, seller terms,
support; 22 pages, prose-only).

**Every hybrid beat BM25 alone on Recall@1 and on MRR — four models out of four, zero
counterexamples.**

| Retriever | R@1 | R@3 | R@5 | R@10 | MRR |
|---|---|---|---|---|---|
| BM25 only (baseline) | 53% | 79% | 89% | 95% | 0.664 |
| HYBRID: BM25 + e5-small-v2 | 63% | 79% | 84% | 100% | 0.717 |
| **HYBRID: BM25 + bge-small-en-v1.5** ✅ | **58%** | **84%** | **89%** | **95%** | **0.703** |
| HYBRID: BM25 + gte-small | 58% | 79% | 89% | 95% | 0.687 |
| HYBRID: BM25 + all-MiniLM-L6-v2 | 63% | 74% | 89% | 89% | 0.712 |

**Selected: BM25 + `bge-small-en-v1.5`** — the only configuration non-negative on every
metric (the others each regress somewhere: e5 loses R@5, MiniLM loses R@3). R@1 is the
metric that matters here: a higher top-1 hit rate means fewer chunks passed to the LLM,
which is a direct latency and RAM saving on CPU-only 8 GB hardware. Full rationale,
per-stratum results and the methodological caveats are in
[`docs/DECISION-002-retrieval-architecture.md`](docs/DECISION-002-retrieval-architecture.md).

`n = 19 questions` — thin. The result rests on **consistency across four independently
trained models**, not the size of any single gap. A later re-run of this same locked
config against an expanded 35-question set is in
[`benchmarks/retrieval_n35/README.md`](benchmarks/retrieval_n35/README.md) — numbers
above are unrevised n=19; do not conflate them with the n=35 figures.

## Generation result

Three grounding prompts (v1, v2, v3) benchmarked against 35 answerable questions + 6
"unanswerable" abstention probes, `k=3`, Qwen2.5-3B-Instruct via local Ollama,
`temperature=0 seed=42`, fully offline. Graded by a deterministic, model-free checker
(Layer A) applied identically to all three.

| Metric | v1 | v2 | v3 |
|---|---|---|---|
| Answerable pass | 16/35 (45.7%) | 6/35 (17.1%) | **25/35 (71.4%)** |
| Laundered answers | 19 | 23 | **0** |
| Abstention probes correct | 6/6 | 6/6 | **6/6** |

**v2 was a large, diagnosed regression** — a prompt structure that made the abstain branch
more salient than the answer branch made abstention the model's default, and 20 of its 27
abstentions "laundered" a correct answer under a false refusal label. v3 fixed it by removing
the general-knowledge note and inverting the salience. Conditioned on retrieval actually
supplying the gold chunk, v3 generation is correct on **27/28 (96%)**, and the abstention
safety property held with zero fabrication events across all three prompts (18 probe
evaluations). **[DECISION-004](docs/DECISION-004-grounding-prompt.md) locks the v3 prompt.**
Full method, per-stratum breakdown, and stated limitations (Layer A is a heuristic, not a
truth oracle; hand-validated 2026-07-23 against a 13-item adversarial sample — 9/10 sampled
PASSes confirmed correct, 1 confirmed ungrounded — see
[`R5_validation_result.md`](benchmarks/generation/R5_validation_result.md)) in
[`benchmarks/generation/README.md`](benchmarks/generation/README.md).

## Evaluation evidence, at a glance

| Study | n | Headline | Record |
|---|---|---|---|
| Retrieval, locked config | 19 | R@1 58% / R@5 89% / MRR 0.703 | [DECISION-002](docs/DECISION-002-retrieval-architecture.md) |
| Retrieval, same config, bigger set | 35 | R@1 60% / R@5 83% / MRR 0.704; prose R@5 62% | [`benchmarks/retrieval_n35/`](benchmarks/retrieval_n35/README.md) |
| Reranker (rejected on n=19) | 19 | rejected — no failure shape it can fix | [DECISION-003](docs/DECISION-003-reranker.md) |
| Reranker (reopened on n=35) | 35 | R@5 83%→89%, prose 62%→75%, 2 regressions — **decided: not shipped in v1** | [DECISION-005](docs/DECISION-005-reranker-reopen.md) |
| Generation (grounding prompt) | 35 + 6 probes | v3 locked: 71.4% pass, 27/28 (96%) conditioned on retrieval, 0 fabrications | [DECISION-004](docs/DECISION-004-grounding-prompt.md) |

## Quickstart

```bash
make setup   # install deps + vendor the tokenizer
make all     # ingest → label → benchmark → verify
```

`make setup` installs `requirements-bench.txt` (which pulls in PyTorch for benchmarking)
and vendors the tokenizer. To run only the offline pipeline (ingest / label / verify)
without PyTorch, `pip install -r requirements.txt` is enough.

Individual stages:

```bash
make ingest   # chunk data/raw/*.pdf  -> benchmarks/chunks_sme.txt (+ corpus fingerprint)
make label    # auto-label by proof   -> data/questions/questions_sme_auto.json
make bench    # run the retrieval bake-off (needs requirements-bench.txt)
make verify   # assert the corpus fingerprint matches the gold-label set
```

## Reproducibility (why this is enforced, not assumed)

Chunk IDs are **positional**, and gold labels are `(question → chunk_id)` pairs — so if
ingestion drifts by even one chunk, every label silently points at the wrong text. This
already happened once: an unpinned tokenizer produced 47 chunks on one machine and 57 on
another, invalidating an entire benchmark. Three guardrails prevent a repeat:

1. **Pinned dependencies.** `requirements.txt` is pinned exactly. `sentence-transformers`
   (≈800 MB of PyTorch) lives in `requirements-bench.txt` and is a **benchmark-only**
   dependency — production uses ONNX. Do not merge the two files.
2. **Vendored tokenizer.** `src/tokenizer.json` is committed. Without it the pipeline
   would fetch from the HuggingFace Hub at runtime, breaking the offline guarantee.
3. **Corpus fingerprint gate.** `scripts/verify_reproducibility.py` refuses to proceed
   unless the freshly-ingested corpus hashes to the value the labels were made against.

The expected corpus is **47 chunks**; auto-labelling yields **19/22 proven, 3 abstained**
(Q11, Q18, Q20 — the labeller abstains rather than guess). See
[`docs/CONCEPTS.md`](docs/CONCEPTS.md) for the concepts behind BM25, dense retrieval, RRF,
and abstaining gold labels.

## Layout

```
src/          retriever, chunker, autolabeller, eval harnesses (retrieval n=19/n=35,
              reranker, generation), extraction/
scripts/      tokenizer vendoring + reproducibility gate
data/         raw/ (source PDFs) and questions/ (draft + gold sets: 19q, 35q, abstention
              probes)
benchmarks/   chunk dumps, result logs, review files; generation/ and retrieval_n35/
              hold the versioned eval results
docs/         DECISION-002 (retrieval), DECISION-003 (reranker, n=19), DECISION-004
              (grounding prompt), DECISION-005 (reranker reopened, n=35) + CONCEPTS
              explainer + session reports
tests/        extraction regression suite (pytest)
```

## Chunkless variant

This repo now also contains a **chunkless** RAG path that navigates document
structure (sections/pages) instead of fixed-size chunks:

```bash
# ingest as sections
chunkless-rag ingest data/raw/*.pdf --chunkless --index-dir ~/.chunkless/index
# ask (reasoning navigator over the section outline)
chunkless-rag ask "What is the return window?" --chunkless --index-dir ~/.chunkless/index
# BM25-only, no LLM navigator
chunkless-rag ask "..." --chunkless --no-navigator --index-dir ~/.chunkless/index
```

See `WORK_PLAN.md` for the full adaptation plan and `papers/` for the grounding
literature (DeepRead, Adaptive Query Routing, Lost in the Middle).

## License

MIT — see [`LICENSE`](LICENSE). The challenge requires open source. Note that model
weights (Qwen, the embedding models) carry their own separate licences.
