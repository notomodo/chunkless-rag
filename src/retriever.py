#!/usr/bin/env python3
"""Hybrid retriever — BM25 (lexical) + dense (semantic), fused by RRF.

WHY HYBRID
==========
The bake-off (bake_off.txt, 2026-07-13) measured a fact that a single retriever
cannot exploit: BM25 and the dense models fail on DIFFERENT questions.

    BM25          R@5 89%   fails: Q02, Q11
    e5-small-v2   R@5 78%   fails: Q02, Q05, Q11, Q12

BM25 alone is the best SINGLE retriever on this corpus. That is not the
interesting number. The interesting number is the union: Q05 and Q12 are BM25
wins, and the dense models are the only thing that could plausibly recover Q02
(period alias: "first half of 2024" vs "H1 2024"). Fusing them targets the
union, not the max.

Note the honest caveat: on THIS corpus BM25 outranks every dense model. If the
fused retriever does not beat BM25's 89% / 0.704, hybrid is not earning its RAM
and we ship BM25 alone. This module exists to make that measurable.

WHY RECIPROCAL RANK FUSION (RRF), NOT SCORE NORMALISATION
---------------------------------------------------------
BM25 scores are unbounded TF-IDF sums; cosine similarities are bounded [-1, 1].
Combining them requires normalising two distributions whose shapes differ per
query — min-max normalisation is notoriously unstable when one list has a single
runaway top hit. RRF sidesteps this entirely by discarding scores and fusing
RANKS:

    RRF(d) = sum over retrievers r of  1 / (k + rank_r(d))

k=60 is the constant from Cormack et al. (SIGIR 2009), "Reciprocal Rank Fusion
outperforms Condorcet and Individual Rank Learning Methods". It damps the
influence of top-1 so a single confident-but-wrong retriever cannot dominate.

It is also parameter-light (one constant), needs no training, and costs nothing.

WHY BM25 IS HAND-ROLLED
-----------------------
`rank_bm25` is a 200-line pure-Python package. Vendoring the ~40 lines we need
removes a dependency from an offline, reproducible, supply-chain-audited build.
The formula is Robertson/Sparck-Jones BM25 with the standard k1=1.5, b=0.75.

DEPENDENCIES
------------
BM25 half:   stdlib only.
Dense half:  pluggable. `SentenceTransformerEncoder` for benchmarking,
             `OnnxEncoder` for shipping. The retriever does not care which —
             it takes any object with `.encode(list[str]) -> np.ndarray`.
             numpy is the only hard requirement.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Protocol, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------

# Three token classes, and the ORDER of alternation matters:
#
#   1. [a-z]+\d+   period codes: h1, q2, fy25. MUST come first, or the [a-z]+
#                  branch matches "h" and leaves "1" as a separate token --
#                  which destroys every period reference in a financial table.
#                  Measured: this bug made "H1" unfindable across the entire
#                  index. It is not cosmetic.
#   2. \d[\d,\.]*  numeric literals: 1,505,398 and 51.5 stay whole. Without
#                  this, figures become digit confetti and exact-figure lookup
#                  degrades to matching the token "5".
#   3. [a-z]+      ordinary words.
_TOKEN_RE = re.compile(r"[a-z]+\d+|\d[\d,\.]*|[a-z]+")

# Vocabulary bridging. BM25 cannot match a query term against a document term it
# has never seen -- it has no notion of synonymy. Financial documents write "H1"
# where users say "first half"; the source spells "fiber" where Ugandan English
# writes "fibre". These are not model failures, they are vocabulary gaps, and
# the honest fix at the lexical layer is an explicit, auditable alias table
# rather than pretending an embedder will rescue it.
#
# This is DELIBERATELY small and domain-specific. It is not a general thesaurus.
# Every entry earns its place by fixing a measured failure. Expanding it
# speculatively is how lexical systems rot.
_ALIASES: dict[str, tuple[str, ...]] = {
    "h1": ("first", "half", "interim", "sixmonth"),
    "h2": ("second", "half"),
    "q1": ("first", "quarter"),
    "q2": ("second", "quarter"),
    "q3": ("third", "quarter"),
    "q4": ("fourth", "quarter"),
    "fy": ("full", "year", "annual"),
    "yoy": ("growth", "grew", "increase"),
    "pat": ("profit",),
    "pbt": ("profit",),
    "capex": ("capital", "expenditure"),
    "fiber": ("fibre",),
    "fibre": ("fiber",),
    "ebitda": ("earnings",),
}


def tokenize(text: str, expand: bool = False) -> list[str]:
    """Lowercase; keep period codes and numeric literals intact.

    `expand=True` additionally emits alias tokens. Apply it to DOCUMENTS, not
    queries: expanding the document side means a chunk containing "H1" is also
    indexed under "first"/"half", so the user's natural phrasing finds it.
    Expanding the query side instead would inflate every query with terms that
    may not be intended, and inflate IDF statistics unpredictably.
    """
    toks = _TOKEN_RE.findall(text.lower())
    if not expand:
        return toks
    out = list(toks)
    for t in toks:
        out.extend(_ALIASES.get(t, ()))
    return out


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------


class BM25:
    """Okapi BM25. Pure stdlib.

    k1 controls term-frequency saturation (how fast repeated terms stop helping).
    b  controls length normalisation (0 = none, 1 = full).
    The 1.5 / 0.75 defaults are the standard Robertson values and are not worth
    tuning on an 18-question set — that would be fitting noise.
    """

    def __init__(
        self,
        corpus: Sequence[str],
        k1: float = 1.5,
        b: float = 0.75,
        expand_docs: bool = True,
    ) -> None:
        self.k1 = k1
        self.b = b
        # Documents are indexed WITH aliases; queries are tokenized literally.
        # See tokenize() for why the asymmetry is the right way round.
        self.docs: list[list[str]] = [tokenize(d, expand=expand_docs) for d in corpus]
        self.n = len(self.docs)
        self.doc_len = np.array([len(d) for d in self.docs], dtype=np.float32)
        self.avgdl = float(self.doc_len.mean()) if self.n else 0.0

        self.tf: list[Counter[str]] = [Counter(d) for d in self.docs]
        df: Counter[str] = Counter()
        for d in self.docs:
            df.update(set(d))

        # Robertson-Sparck-Jones IDF with the +0.5 smoothing and a floor.
        # Without the floor, a term appearing in >half the docs gets NEGATIVE
        # idf and actively penalises documents that contain it — which for a
        # corpus where "revenue" is in most chunks would be catastrophic.
        self.idf: dict[str, float] = {
            term: max(
                1e-6, math.log((self.n - freq + 0.5) / (freq + 0.5) + 1.0)
            )
            for term, freq in df.items()
        }

    def scores(self, query: str) -> np.ndarray:
        """Return a score for every document. Higher is better."""
        q_terms = tokenize(query)
        out = np.zeros(self.n, dtype=np.float32)
        for term in q_terms:
            idf = self.idf.get(term)
            if idf is None:
                continue  # OOV term contributes nothing
            for i, tf_i in enumerate(self.tf):
                f = tf_i.get(term, 0)
                if not f:
                    continue
                denom = f + self.k1 * (
                    1 - self.b + self.b * self.doc_len[i] / self.avgdl
                )
                out[i] += idf * (f * (self.k1 + 1)) / denom
        return out


# ---------------------------------------------------------------------------
# Dense encoders (pluggable)
# ---------------------------------------------------------------------------


class Encoder(Protocol):
    """Anything that turns text into L2-normalised vectors.

    Deliberately minimal. The retriever must not know or care whether the
    vectors came from sentence-transformers, ONNX Runtime, or a stub.
    """

    def encode(self, texts: Sequence[str], is_query: bool = False) -> np.ndarray:
        ...


# Asymmetric-prefix models. Omitting the prefix measurably degrades retrieval —
# these models were trained with them and benchmarking without is meaningless.
_PREFIXES: dict[str, tuple[str, str]] = {
    "intfloat/e5": ("query: ", "passage: "),
    "BAAI/bge": ("Represent this sentence for searching relevant passages: ", ""),
}


def _prefix_for(model_name: str) -> tuple[str, str]:
    for key, pair in _PREFIXES.items():
        if model_name.startswith(key):
            return pair
    return ("", "")


class SentenceTransformerEncoder:
    """Benchmark-time encoder. NOT the shipping path — sentence-transformers
    pulls in torch (~800 MB installed), which we will not ship on an 8 GB
    offline target. Use OnnxEncoder for deployment; this exists so the
    benchmark numbers are comparable to the bake-off."""

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer  # lazy: optional dep

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.q_prefix, self.p_prefix = _prefix_for(model_name)
        self.max_seq = self.model.max_seq_length

    def encode(self, texts: Sequence[str], is_query: bool = False) -> np.ndarray:
        prefix = self.q_prefix if is_query else self.p_prefix
        return self.model.encode(
            [prefix + t for t in texts],
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)


#: The vendored tokenizer that ships with the repo, committed alongside this
#: module. Loading it is what keeps the shipping path OFFLINE.
VENDORED_TOKENIZER = Path(__file__).parent / "tokenizer.json"


def _load_vendored_tokenizer(tokenizer_path: str | Path | None = None):
    """Load the tokenizer for the SHIPPING path from a vendored file — OFFLINE.

    The headline guarantee of this project is "no API calls, no network at
    runtime". `Tokenizer.from_pretrained()` reaches the Hugging Face Hub, so it
    must NEVER be on the shipping path; we load a committed `tokenizer.json`
    instead (the same file `ingest_sme.py` uses).

    CRITICAL — this deliberately has NO Hub fallback. `ingest_sme.py` warns and
    falls back to `from_pretrained()` when the vendored file is missing; for the
    shipping encoder that fallback IS the offline violation we are eliminating.
    A missing vendored file is a hard error, not a silent network fetch. Fail
    hard; do not degrade.
    """
    path = Path(tokenizer_path) if tokenizer_path is not None else VENDORED_TOKENIZER
    if not path.exists():
        raise FileNotFoundError(
            f"Vendored tokenizer not found at {path}. The shipping encoder is "
            f"offline by contract and will NOT fetch from the Hugging Face Hub. "
            f"Run `python scripts/vendor_tokenizer.py` to vendor it, then commit "
            f"the resulting src/tokenizer.json."
        )
    from tokenizers import Tokenizer  # local import: optional dep, no torch

    return Tokenizer.from_file(str(path))


class OnnxEncoder:
    """Shipping encoder. onnxruntime + tokenizers only — no torch.

    Pools the last hidden state with **CLS pooling** (`hidden[:, 0]`), then
    L2-normalises. This must match whatever pooling sentence-transformers applied
    during the bake-off, because that is what the benchmarked numbers describe —
    a different pooling here is a silent retrieval-quality regression.

    Pooling is per-model, NOT uniform across the shortlist:
      - bge-small-en-v1.5  -> CLS   (1_Pooling/config.json: pooling_mode_cls_token=true)
      - e5-small-v2, gte-small, all-MiniLM-L6-v2 -> mean
    We ship bge (DECISION-002), so this encoder does CLS. Verified: CLS reproduces
    the SentenceTransformerEncoder vectors to <1e-6 max abs diff on the probe set;
    mean pooling diverged to cosine ~0.93 (tests/test_onnx_parity.py locks this).
    A prior version mean-pooled unconditionally and claimed "that is what
    sentence-transformers does for every model in the shortlist" — false for bge.

    OFFLINE by contract: the tokenizer is loaded from the vendored
    `tokenizer.json`, never `from_pretrained()` (which hits the Hub). See
    `_load_vendored_tokenizer`. `tokenizer_name` is retained purely to select the
    model's asymmetric query/passage prefixes; it no longer triggers a download.
    """

    def __init__(
        self,
        onnx_path: str,
        tokenizer_name: str,
        model_name: str = "",
        tokenizer_path: str | Path | None = None,
    ) -> None:
        import onnxruntime as ort

        self.session = ort.InferenceSession(
            onnx_path, providers=["CPUExecutionProvider"]
        )
        self.tok = _load_vendored_tokenizer(tokenizer_path)
        self.q_prefix, self.p_prefix = _prefix_for(model_name or tokenizer_name)
        self._inputs = {i.name for i in self.session.get_inputs()}

    def encode(self, texts: Sequence[str], is_query: bool = False) -> np.ndarray:
        prefix = self.q_prefix if is_query else self.p_prefix
        # Truncate very long inputs to a safe token budget so a single long
        # section cannot overflow the model's context window. Char-based
        # truncation is a conservative proxy; the tokenizer re-tokenises below.
        safe = [ (t[:1500] if not is_query else t[:512]) for t in texts ]
        encs = self.tok.encode_batch([prefix + t for t in safe])
        maxlen = max(len(e.ids) for e in encs)

        ids = np.zeros((len(encs), maxlen), dtype=np.int64)
        mask = np.zeros((len(encs), maxlen), dtype=np.int64)
        for i, e in enumerate(encs):
            ids[i, : len(e.ids)] = e.ids
            mask[i, : len(e.attention_mask)] = e.attention_mask

        feed = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self._inputs:
            feed["token_type_ids"] = np.zeros_like(ids)

        hidden = self.session.run(None, feed)[0]  # (B, T, H)

        # CLS pooling: bge represents the sequence in position 0 (the [CLS] token),
        # not the mean over tokens. The attention mask is still needed above to
        # build the model input, but plays no part in pooling for a CLS model.
        pooled = hidden[:, 0]
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        return (pooled / np.clip(norms, 1e-9, None)).astype(np.float32)


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------


def rrf_fuse(
    rankings: Iterable[Sequence[int]], k: float = 60.0, weights: Sequence[float] | None = None
) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion.

    `rankings` is one ranked list of doc-ids per retriever, best-first.
    Returns [(doc_id, fused_score)] sorted best-first.

    Documents absent from a retriever's list simply contribute nothing from it —
    they are not penalised. This is what makes RRF robust to retrievers with
    very different score scales and very different recall profiles.
    """
    rankings = list(rankings)
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("weights must match number of rankings")

    fused: dict[int, float] = {}
    for w, ranking in zip(weights, rankings):
        for rank, doc_id in enumerate(ranking, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + w / (k + rank)

    return sorted(fused.items(), key=lambda kv: -kv[1])


# ---------------------------------------------------------------------------
# The retriever
# ---------------------------------------------------------------------------


@dataclass
class Hit:
    chunk_id: int
    score: float
    text: str
    meta: dict = field(default_factory=dict)


class HybridRetriever:
    """BM25 + dense, fused by RRF.

    Set `encoder=None` to get a pure-BM25 retriever with the identical API —
    this is the honest fallback if the dense half fails to earn its RAM, and it
    means the application layer never needs to know which mode it is in.
    """

    def __init__(
        self,
        chunks: Sequence[str],
        encoder: Encoder | None = None,
        metas: Sequence[dict] | None = None,
        rrf_k: float = 60.0,
        candidate_depth: int = 50,
        weights: tuple[float, float] = (1.0, 1.0),
    ) -> None:
        self.chunks = list(chunks)
        self.metas = list(metas) if metas else [{} for _ in chunks]
        self.encoder = encoder
        self.rrf_k = rrf_k
        self.weights = weights

        # How deep each retriever's list goes before fusion. Too shallow and a
        # document ranked 30th by BM25 but 2nd by dense never enters the fusion
        # at all — which defeats the purpose. 50 is generous for a 37-chunk
        # corpus and still trivial at 10k chunks.
        self.candidate_depth = min(candidate_depth, len(self.chunks))

        self.bm25 = BM25(self.chunks)
        self.doc_vecs: np.ndarray | None = None
        if encoder is not None:
            self.doc_vecs = encoder.encode(self.chunks, is_query=False)

    # -- individual retrievers ------------------------------------------------

    def _bm25_ranking(self, query: str, depth: int) -> list[int]:
        scores = self.bm25.scores(query)
        # argsort is ascending; take the tail and reverse.
        idx = np.argsort(scores)[::-1][:depth]
        # Drop zero-score docs: a doc sharing no query term is not a "rank 40
        # candidate", it is a non-candidate, and injecting it into RRF adds noise.
        return [int(i) for i in idx if scores[i] > 0.0]

    def _dense_ranking(self, query: str, depth: int) -> list[int]:
        if self.doc_vecs is None:
            return []
        q = self.encoder.encode([query], is_query=True)[0]
        sims = self.doc_vecs @ q  # both L2-normalised => cosine
        idx = np.argsort(sims)[::-1][:depth]
        return [int(i) for i in idx]

    # -- public API -----------------------------------------------------------

    def search(self, query: str, top_k: int = 5) -> list[Hit]:
        depth = self.candidate_depth
        bm = self._bm25_ranking(query, depth)

        if self.doc_vecs is None:
            ranked = [(d, 1.0 / (self.rrf_k + r)) for r, d in enumerate(bm, 1)]
        else:
            dn = self._dense_ranking(query, depth)
            ranked = rrf_fuse([bm, dn], k=self.rrf_k, weights=list(self.weights))

        return [
            Hit(chunk_id=d, score=s, text=self.chunks[d], meta=self.metas[d])
            for d, s in ranked[:top_k]
        ]
