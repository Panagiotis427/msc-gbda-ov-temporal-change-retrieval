"""Tests for scripts/build_qfabric_labels.py — majority-vote label resolution.

Regression guard for the audit-found bug: a crop with conflicting RQA2 answers
must resolve to the MAJORITY change type, not last-write-wins.
"""
from __future__ import annotations

import json

from scripts.build_qfabric_labels import build


def _rec(loc, day, xoff, yoff, answer):
    name = f"X/{loc}.d{day}.0101201{day}_{xoff}_{yoff}.tif"
    return {"video": [name],
            "conversations": [{"from": "human", "value": "type? [0,0,1,1]"},
                              {"from": "gpt", "value": answer}]}


def test_majority_vote_resolves_conflict(tmp_path):
    rqa2 = tmp_path / "rqa2.json"
    # crop 10_0_0: 2x Residential, 1x Commercial -> majority Residential
    # crop 20_0_0: unanimous Road
    recs = [
        _rec(10, 1, 0, 0, "Residential"),
        _rec(10, 2, 0, 0, "Residential"),
        _rec(10, 3, 0, 0, "Commercial"),
        _rec(20, 1, 0, 0, "Road"),
        _rec(20, 2, 0, 0, "Road"),
    ]
    rqa2.write_text(json.dumps(recs))
    out = tmp_path / "labels.json"
    labels = build(str(rqa2), str(out))
    assert labels["10_0_0"] == "residential"   # majority, NOT last-seen commercial
    assert labels["20_0_0"] == "road"
    # round-trips to disk
    assert json.loads(out.read_text()) == labels


def test_mega_projects_normalised(tmp_path):
    rqa2 = tmp_path / "rqa2.json"
    rqa2.write_text(json.dumps([_rec(30, 1, 256, 512, "Mega Projects")]))
    labels = build(str(rqa2), str(tmp_path / "o.json"))
    assert labels["30_256_512"] == "mega_projects"
