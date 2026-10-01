"""Shared statistical helpers for the retrieval-evaluation scripts.

``rand_ap`` is the average precision of a *uniformly random* ranking of ``N``
items with ``R`` relevant — the honest chance baseline for AP (whose expectation
is ~prevalence, not zero; only approximately so, and loosely for small ``N``, which
is why the permutation tests compare against simulated random rankings rather than
against the prevalence). ``scripts/patch_eval`` and
``scripts/significance_audit`` each carried a byte-identical copy of this; this
module is the single source of truth so the permutation baseline can never drift
between them.

NOTE: ``scripts/cv_eval`` deliberately keeps its own permutation routine — it
draws via ``rng.permutation`` rather than ``rng.shuffle``, so swapping it for
this helper would change its RNG draw sequence and perturb the already-committed
``cv_eval__*.json`` results. It is intentionally left separate.
"""
from __future__ import annotations

import numpy as np


def rand_ap(R: int, N: int, rng) -> float:
    """AP of a uniformly random ranking of ``N`` items with ``R`` relevant.

    ``rng`` is a ``numpy`` Generator/RandomState; one ``rng.shuffle`` call is
    consumed per invocation (callers rely on this for reproducible draws).
    """
    rel = np.zeros(N, dtype=bool)
    rel[:R] = True
    rng.shuffle(rel)
    hits = np.cumsum(rel)
    return float((hits / np.arange(1, N + 1))[rel].sum() / R)


def rank_order(scores: np.ndarray, rel: np.ndarray) -> np.ndarray:
    """Indices that rank ``scores`` descending, with deterministic, *pessimistic*
    tie-breaking: among equal scores a non-relevant item is ranked above a
    relevant one (so ties never inflate AP), and the order is fully reproducible.

    Implemented as a stable ``lexsort`` with ``-scores`` as the primary key and the
    relevance flag as the secondary key (0 = non-relevant sorts first within a tie).
    Use this everywhere AP / Recall@K is computed so results never depend on the
    undefined ordering of ``np.argsort`` over tied (e.g. bootstrap-duplicated)
    scores.
    """
    scores = np.asarray(scores, dtype=np.float64)
    rel = np.asarray(rel)
    return np.lexsort((rel.astype(np.int8), -scores))


def perm_p_value(n_ge: int, iters: int) -> float:
    """Unbiased one-sided Monte-Carlo permutation p-value.

    ``n_ge`` = number of null draws with statistic >= observed. The observed
    statistic is itself one realisation of the null, so it must be counted:
    ``(n_ge + 1) / (iters + 1)``. This can never return an impossible ``0.0``.
    """
    return (int(n_ge) + 1) / (int(iters) + 1)


def aoi_folds(aois, n_folds: int, seed: int) -> dict:
    """Deterministic AOI -> fold assignment for leave-one-group-out CV.

    Round-robin over a seeded permutation of the *sorted* AOI ids. Shared by
    ``scripts.cv_eval`` and ``scripts.patch_eval`` so their k-fold partitions are
    provably identical (same seed -> same partition), making their cross-validated
    mAPs directly comparable. Folds are disjoint and cover every AOI.
    """
    rng = np.random.default_rng(seed)
    perm = list(rng.permutation(sorted(aois)))
    return {a: i % n_folds for i, a in enumerate(perm)}


def bh_fdr(pvals) -> np.ndarray:
    """Benjamini-Hochberg FDR-adjusted q-values for ``pvals``.

    Returns q-values aligned to the input order. The raw ``p * m / rank`` is not
    monotone in rank, so the standard step-up enforces monotonicity by taking the
    running minimum from the largest p downward, then caps at 1.
    """
    p = np.asarray(pvals, dtype=np.float64)
    m = p.size
    if m == 0:
        return np.empty(0)
    order = np.argsort(p)
    ranked = p[order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.clip(ranked, 0.0, 1.0)
    return out
