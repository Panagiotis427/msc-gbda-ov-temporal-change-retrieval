"""
Tied change scores must rank in corpus order, every time: the app's results and
the CSV export show exactly this order, so a refactor that drops the stable sort
(say, for ``np.argpartition``) has to fail here. The corpus is large enough (64
pairs) that NumPy's default sort — insertion sort, hence stable, below 16
elements — would reorder the ties.
"""
from __future__ import annotations

import numpy as np

from src.datasets.base import PairKey
from src.embeddings import PairEmbeddingStore
from src.retrieval import ChangeRetriever

_N = 64
# How much each pair gains of the query direction from T1 to T2: many exact ties.
_GAINS = [(0.8, 0.2, 0.5, 0.8, 0.5, 0.2, 0.8, 0.1)[i % 8] for i in range(_N)]


class _AxisTextEncoder:
    """Every query encodes to the first axis."""
    name = "axis"
    embed_dim = 3

    def encode_text(self, texts, batch_size: int = 32) -> np.ndarray:
        return np.array([[1.0, 0.0, 0.0]], dtype=np.float32)


def _store() -> PairEmbeddingStore:
    t1 = np.tile(np.array([0.0, 1.0, 0.0], dtype=np.float32), (_N, 1))
    t2 = np.array([[g, np.sqrt(1.0 - g * g), 0.0] for g in _GAINS], dtype=np.float32)
    pairs = [PairKey(f"loc{i:02d}", "2020-01-01", "2020-03-01") for i in range(_N)]
    return PairEmbeddingStore(dataset_name="toy", encoder_name="axis", embed_dim=3,
                              pairs=pairs, f_t1=t1, f_t2=t2)


def test_tied_scores_rank_in_corpus_order():
    retr = ChangeRetriever(_store(), _AxisTextEncoder())
    scores = retr.score_all("any change", approach="zero_shot")
    assert len(set(np.round(scores, 6))) == 4  # four distinct gains, the rest ties
    # Python's sort is stable: descending score, ties by corpus index.
    expected = sorted(range(_N), key=lambda i: -scores[i])[:20]
    for _ in range(3):
        hits = retr.search("any change", approach="zero_shot", top_k=20)
        assert [h.pair.location_id for h in hits] == [f"loc{i:02d}" for i in expected]
        assert [h.rank for h in hits] == list(range(20))
