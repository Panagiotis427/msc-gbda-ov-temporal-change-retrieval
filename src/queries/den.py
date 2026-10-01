"""
Dynamic EarthNet query set + relevance predicates over DEN's 7 LULC classes.

Self-registers under the dataset name ``"dynamic_earthnet"`` when this module
is imported (``src.queries`` imports it at package load).
"""
from __future__ import annotations

from typing import List

from src.benchmark import Query, _gained, _lost, _t1, _transition
from . import register_queries

# DEN LULC class names (must match src.datasets.dynamic_earthnet.CLASS_NAMES).
IMPERVIOUS = "impervious_surface"
AGRI = "agriculture"
FOREST = "forest_and_other_vegetation"
WETLANDS = "wetlands"
SOIL = "soil"
WATER = "water"
SNOW = "snow_and_ice"


QUERIES = [
    Query("new buildings constructed on former agricultural land",
          "permanent", _transition(AGRI, IMPERVIOUS)),
    Query("urban expansion replacing vegetation",
          "permanent", lambda lb: _transition(dst=IMPERVIOUS)(lb)
          and _t1(lb) in (AGRI, FOREST)),
    Query("deforestation, forest cleared to bare soil",
          "permanent", _transition(FOREST, SOIL)),
    Query("forest loss",
          "permanent", lambda lb: _t1(lb) == FOREST and not (lb is None or lb.stable)
          and lb.dominant_t2_class not in (FOREST, None)),
    Query("new water body or flooding",
          "permanent", _transition(dst=WATER)),
    Query("bare soil or land cleared",
          "permanent", _transition(dst=SOIL)),
    Query("seasonal snow melting away",
          "seasonal", _transition(src=SNOW)),
    Query("agricultural land converted to wetland or marsh",
          "permanent", _transition(AGRI, WETLANDS)),
    Query("wetland drained and turned into farmland",
          "permanent", _transition(WETLANDS, AGRI)),
    Query("land turning into wetland",
          "permanent", _transition(dst=WETLANDS)),
]


register_queries("dynamic_earthnet", QUERIES)
# Native 3 m Planet-Fusion DEN uses the same 7 LULC classes and PairLabel schema,
# so it shares this query set (resolved by dataset.name in run_benchmark).
register_queries("dynamic_earthnet_planet", QUERIES)


# ---------------------------------------------------------------------------
# A-priori change *geometry* per query — for the query-type-gated global/patch
# hybrid (report §8.3). Tagged from the change's spatial extent ALONE,
# never fit to the results (no peeking): compact / point-like features whose
# signal lives in a few patches -> "localised" (patch_top3 wins, report §8.1); broad /
# areal cover change that moves the whole-tile embedding -> "diffuse" (global
# Delta wins, report §8.1). Borderline footprints (deforestation, bare soil) are tagged
# by their typical extent. Consumed by
# ``scripts/patch_eval.py --approach gated``; keyed by the shared query text
# (identical across QUERIES and frac_queries).
# ---------------------------------------------------------------------------
DEN_QUERY_GEOMETRY = {
    "new buildings constructed on former agricultural land": "localised",
    "urban expansion replacing vegetation": "localised",
    "deforestation, forest cleared to bare soil": "localised",
    "new water body or flooding": "localised",
    "forest loss": "diffuse",
    "bare soil or land cleared": "diffuse",
    "seasonal snow melting away": "diffuse",
    "agricultural land converted to wetland or marsh": "diffuse",
    "wetland drained and turned into farmland": "diffuse",
    "land turning into wetland": "diffuse",
}


def frac_queries(thresh: float = 0.05) -> List[Query]:
    """Fraction-based relevance variant of :data:`QUERIES` (same 10 texts).

    A pair is relevant when the target class gains/loses >= ``thresh`` of valid
    pixels — capturing localised change the dominant-class-flip predicates miss
    (see ``benchmark._gained`` / ``_lost``). Directional pairs use the
    discriminating side (e.g. wetland *drained* = wetlands lost). Used by
    ``scripts/cv_eval.py --relevance fraction``; not registered as the default
    so the committed dominant-flip numbers stay reproducible.
    """
    return [
        Query("new buildings constructed on former agricultural land",
              "permanent", _gained(IMPERVIOUS, thresh)),
        Query("urban expansion replacing vegetation",
              "permanent", _gained(IMPERVIOUS, thresh)),
        Query("deforestation, forest cleared to bare soil",
              "permanent", _lost(FOREST, thresh)),
        Query("forest loss",
              "permanent", _lost(FOREST, thresh)),
        Query("new water body or flooding",
              "permanent", _gained(WATER, thresh)),
        Query("bare soil or land cleared",
              "permanent", _gained(SOIL, thresh)),
        Query("seasonal snow melting away",
              "seasonal", _lost(SNOW, thresh)),
        Query("agricultural land converted to wetland or marsh",
              "permanent", _gained(WETLANDS, thresh)),
        Query("wetland drained and turned into farmland",
              "permanent", _lost(WETLANDS, thresh)),
        Query("land turning into wetland",
              "permanent", _gained(WETLANDS, thresh)),
    ]
