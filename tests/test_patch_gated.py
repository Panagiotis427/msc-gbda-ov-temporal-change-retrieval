"""Unit tests for the query-type-gated global/patch hybrid (report §8.3).

Pure-numpy, CPU-only — no encoder weights, no dataset, no cache. Verifies the
routing in ``scripts.patch_eval._scores`` (diffuse -> global Δ, localised ->
patch_top3) and that the a-priori geometry map covers every DEN query.
"""
import numpy as np

from scripts.patch_eval import _patch_score, _scores
from src.queries.den import DEN_QUERY_GEOMETRY, frac_queries


def _toy():
    rng = np.random.default_rng(0)
    d = 8
    P1 = rng.standard_normal((4, 6, d)).astype(np.float32)
    P2 = rng.standard_normal((4, 6, d)).astype(np.float32)
    G1 = rng.standard_normal((4, d)).astype(np.float32)
    G2 = rng.standard_normal((4, d)).astype(np.float32)
    t = rng.standard_normal(d).astype(np.float32)
    return P1, P2, G1, G2, t


def test_gated_localised_equals_patch_top3():
    P1, P2, G1, G2, t = _toy()
    got = _scores(P1, P2, t, "gated", G1, G2, geom="localised")
    np.testing.assert_allclose(got, _patch_score(P1, P2, t, "patch_top3"))


def test_gated_diffuse_equals_global_delta():
    P1, P2, G1, G2, t = _toy()
    got = _scores(P1, P2, t, "gated", G1, G2, geom="diffuse")
    np.testing.assert_allclose(got, (G2 @ t) - (G1 @ t))


def test_gated_unknown_geom_falls_back_to_patch():
    # A query with no geometry tag must not crash — defaults to the localised scorer.
    P1, P2, G1, G2, t = _toy()
    got = _scores(P1, P2, t, "gated", G1, G2, geom=None)
    np.testing.assert_allclose(got, _patch_score(P1, P2, t, "patch_top3"))


def test_geometry_map_covers_every_query_with_valid_tags():
    texts = {q.text for q in frac_queries()}
    assert texts <= set(DEN_QUERY_GEOMETRY), "every frac_query needs a geometry tag"
    assert set(DEN_QUERY_GEOMETRY.values()) <= {"diffuse", "localised"}
