"""
Value-level tests for the core retrieval metric `_average_precision`.

mAP is the headline metric of every report §8 table, but the AP formula was previously
exercised only through `run_benchmark` on a perfect-ranking fixture (so the
assertions were trivial). These pin the AP value on imperfect rankings.

`_average_precision(rel)` takes a boolean array already in ranked order.
"""
import numpy as np

from src.benchmark import _average_precision


def test_average_precision_perfect_and_empty():
    assert _average_precision(np.array([True, True])) == 1.0
    assert _average_precision(np.array([True, True, False, False])) == 1.0
    assert _average_precision(np.array([False, False])) == 0.0          # no positives
    assert _average_precision(np.array([], dtype=bool)) == 0.0


def test_average_precision_imperfect_rankings():
    # [1,0,1,0]: precisions at hits = 1/1 and 2/3 -> (1 + 0.6667)/2 = 0.8333
    ap = _average_precision(np.array([True, False, True, False]))
    assert abs(ap - 0.8333333) < 1e-6

    # [0,0,1,1]: precisions at hits = 1/3 and 2/4 -> (0.3333 + 0.5)/2 = 0.4167
    ap2 = _average_precision(np.array([False, False, True, True]))
    assert abs(ap2 - 0.4166667) < 1e-6

    # single relevant item last of five: precision 1/5 -> AP 0.2
    ap3 = _average_precision(np.array([False, False, False, False, True]))
    assert abs(ap3 - 0.2) < 1e-6
