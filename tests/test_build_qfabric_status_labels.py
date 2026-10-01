"""Tests for scripts/build_qfabric_status_labels.py — RQA5 status index.

Covers: image-index parsing ("Image N" + ordinals), status-class matching,
the Image-N -> frame-N join, majority vote per (crop, day), nested output,
and the temporal-task filter.
"""
from __future__ import annotations

import json

from scripts.build_qfabric_status_labels import (
    _image_index, _match_status, build,
)


def _vid(loc, day, xoff, yoff):
    return f"X/QFabric/{loc}.d{day}.0101201{day}_{xoff}_{yoff}.tif"


def _rec(frames, image_n, answer, task="region_based_temporal_question_answering"):
    q = f"<video>\nWhat is the development status of this region [0,0,1,1] in Image {image_n}?"
    return {"task": task, "video": frames,
            "conversations": [{"from": "human", "value": q},
                              {"from": "gpt", "value": answer}]}


def test_image_index_cardinal_and_ordinal():
    assert _image_index("status in Image 1?") == 1
    assert _image_index("status in Image 10?") == 10
    assert _image_index("status of this region in the third image?") == 3
    assert _image_index("no index here") is None


def test_match_status_slugs():
    assert _match_status("Land Cleared") == "land_cleared"
    assert _match_status("Construction Done") == "construction_done"
    assert _match_status("operational") == "operational"          # case-insensitive
    assert _match_status("Prior Construction") == "prior_construction"
    assert _match_status("Banana") is None


def test_image_n_join_and_nested_output(tmp_path):
    frames = [_vid(100, 1, 0, 256), _vid(100, 2, 0, 256), _vid(100, 3, 0, 256)]
    recs = [
        _rec(frames, 1, "Greenland"),
        _rec(frames, 2, "Land Cleared"),
        _rec(frames, 3, "Construction Done"),
    ]
    out = build_from(recs, tmp_path)
    assert out["100_0_256"] == {"d1": "greenland", "d2": "land_cleared",
                                "d3": "construction_done"}


def test_image_zero_is_skipped_not_read_as_the_last_frame(tmp_path):
    frames = [_vid(150, 1, 0, 0), _vid(150, 2, 0, 0)]
    recs = [
        _rec(frames, 0, "Greenland"),        # "Image 0" is no frame (1-based); frames[-1] would be d2
        _rec(frames, 1, "Land Cleared"),
    ]
    out = build_from(recs, tmp_path)
    assert out["150_0_0"] == {"d1": "land_cleared"}


def test_majority_vote_per_timepoint(tmp_path):
    frames = [_vid(200, 1, 0, 0), _vid(200, 2, 0, 0)]
    recs = [
        _rec(frames, 1, "Land Cleared"),
        _rec(frames, 1, "Land Cleared"),
        _rec(frames, 1, "Greenland"),      # minority for (200,d1)
        _rec(frames, 2, "Construction Done"),
    ]
    out = build_from(recs, tmp_path)
    assert out["200_0_0"]["d1"] == "land_cleared"   # majority, not last-seen
    assert out["200_0_0"]["d2"] == "construction_done"


def test_non_temporal_task_ignored(tmp_path):
    frames = [_vid(300, 1, 0, 0), _vid(300, 2, 0, 0)]
    recs = [
        _rec(frames, 1, "Commercial", task="region_based_question_answering"),
        _rec(frames, 1, "Land Cleared"),
        _rec(frames, 2, "Construction Done"),
    ]
    out = build_from(recs, tmp_path)
    # the change-type RQA record is skipped; only the two RTQA statuses survive
    assert out["300_0_0"] == {"d1": "land_cleared", "d2": "construction_done"}


def build_from(recs, tmp_path):
    src = tmp_path / "rqa5.json"
    src.write_text(json.dumps(recs))
    out = tmp_path / "status.json"
    res = build(str(src), str(out))
    assert json.loads(out.read_text()) == res        # round-trips to disk
    return res
