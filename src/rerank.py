"""
Post-retrieval re-ranking using spatial and temporal coherence.

After :meth:`ChangeRetriever.score_all` returns per-pair cosine scores, a
:class:`Reranker` can optionally reorder the top-K results using two
strategies:

* ``diversity``  — greedy location-deduplication: prefers showing results from
  different AOIs before returning to the same location.  Improves result
  coverage without any geographic model.

* ``coherence``  — geographic clustering: boosts pairs whose AOI centroid is
  close to the top-1 result's location (haversine distance).  Useful for
  spatially coherent queries (e.g. "urban expansion in a specific city").

Both strategies are toggleable via the Gradio UI and the CLI ``--rerank``
flag; passing ``strategy=None`` disables re-ranking entirely.
"""
from __future__ import annotations

import json
from math import asin, cos, radians, sin, sqrt
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.datasets.base import PairKey

RERANK_STRATEGIES = ("diversity", "coherence")


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    # Same function as src.geo_filter._haversine_km; keep the two in step.
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return 2 * R * asin(sqrt(max(0.0, min(1.0, a))))


class Reranker:
    """Re-rank retrieval results using spatial coherence or diversity.

    Parameters
    ----------
    metadata_path:
        Path to ``aoi_metadata.json``.  Each key is a ``location_id`` with at
        least ``lat_c`` and ``lon_c`` fields.
    """

    def __init__(self, metadata_path: str | Path) -> None:
        with open(metadata_path) as fh:
            self._meta: Dict[str, dict] = json.load(fh)

    # ------------------------------------------------------------------
    def _centroid(self, location_id: str) -> Optional[Tuple[float, float]]:
        m = self._meta.get(location_id)
        if m is None:
            return None
        lat, lon = m.get("lat_c"), m.get("lon_c")
        if lat is None or lon is None:
            return None
        return float(lat), float(lon)

    # ------------------------------------------------------------------
    def rerank(
        self,
        scores: np.ndarray,
        pairs: List[PairKey],
        top_k: int,
        strategy: str = "diversity",
        geo_weight: float = 0.3,
    ) -> np.ndarray:
        """Return an array of *top_k* indices into *pairs*, re-ranked.

        Parameters
        ----------
        scores:
            Per-pair retrieval scores (higher = better).  May contain ``-inf``
            for pairs masked by a geographic filter.
        pairs:
            Ordered list of :class:`PairKey` aligned with *scores*.
        top_k:
            Number of results to return.
        strategy:
            ``"diversity"`` or ``"coherence"``.
        geo_weight:
            Weight of the geographic term in ``"coherence"`` mode (0–1).

        Returns
        -------
        np.ndarray
            Integer indices into *pairs*, length ≤ *top_k*.
        """
        if strategy == "diversity":
            return self._diversity(scores, pairs, top_k)
        if strategy == "coherence":
            return self._coherence(scores, pairs, top_k, geo_weight)
        raise ValueError(f"Unknown rerank strategy {strategy!r}; use one of {RERANK_STRATEGIES}")

    # ------------------------------------------------------------------
    def _diversity(
        self, scores: np.ndarray, pairs: List[PairKey], top_k: int
    ) -> np.ndarray:
        """Greedy location-diversity re-ranking.

        Iterates pairs in descending score order.  Each new unique location is
        preferred over a repeat; repeats are deferred to fill remaining slots.
        """
        order = list(np.argsort(-scores, kind="stable"))
        seen_locs: set = set()
        result: List[int] = []
        deferred: List[int] = []

        for i in order:
            if not np.isfinite(scores[i]):
                continue
            loc = pairs[i].location_id
            if loc not in seen_locs:
                result.append(i)
                seen_locs.add(loc)
            else:
                deferred.append(i)
            if len(result) >= top_k:
                break

        for i in deferred:
            if len(result) >= top_k:
                break
            result.append(i)

        return np.array(result[:top_k], dtype=int)

    def _coherence(
        self,
        scores: np.ndarray,
        pairs: List[PairKey],
        top_k: int,
        geo_weight: float,
    ) -> np.ndarray:
        """Geographic-coherence re-ranking.

        Boosts pairs geographically close to the top-1 result's centroid.
        Normalized cosine score and proximity are combined linearly:
        ``combined = (1 - w) * norm_score + w * proximity``
        """
        finite_mask = np.isfinite(scores)
        if not finite_mask.any():
            return np.array([], dtype=int)

        # Anchor = highest-scoring finite pair
        masked = np.where(finite_mask, scores, -np.inf)
        top1_idx = int(np.argmax(masked))
        anchor = self._centroid(pairs[top1_idx].location_id)

        if anchor is None:
            # No metadata for top-1 → fall back to default ordering
            return np.argsort(-scores, kind="stable")[:top_k]

        a_lat, a_lon = anchor
        max_dist_km = 5_000.0  # proximity falls linearly to 0 at 5,000 km (about a continent's width)

        prox = np.zeros(len(pairs), dtype=np.float32)
        for i, p in enumerate(pairs):
            if not finite_mask[i]:
                continue
            c = self._centroid(p.location_id)
            if c is None:
                prox[i] = 0.0
            else:
                dist = _haversine_km(a_lat, a_lon, c[0], c[1])
                prox[i] = 1.0 - min(dist / max_dist_km, 1.0)

        # Normalise finite cosine scores to [0, 1]
        finite_scores = scores[finite_mask]
        s_min, s_max = float(finite_scores.min()), float(finite_scores.max())
        span = s_max - s_min if s_max > s_min else 1.0
        norm_scores = np.where(finite_mask, (scores - s_min) / span, 0.0)

        combined = (1.0 - geo_weight) * norm_scores + geo_weight * prox
        # Mask out -inf pairs so they never appear
        combined = np.where(finite_mask, combined, -np.inf)

        return np.argsort(-combined, kind="stable")[:top_k]
