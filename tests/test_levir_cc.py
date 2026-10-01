"""Unit tests for the LEVIR-CC loader + query set, on a tiny synthetic fixture
(no network / no 2.7 GB download needed)."""
import json
import warnings

import numpy as np
import pytest
from PIL import Image

from src.datasets.levir_cc import LevirCCDataset
from src.datasets.registry import build_dataset
from src.queries import get_queries


def _make_fixture(root):
    caps = {"images": [
        {"split": "test", "filename": "test_000000.png", "changeflag": 1,
         "sentences": [{"raw": "many tall buildings and houses are built"},
                       {"raw": "new residential villas appear"}]},
        {"split": "test", "filename": "test_000001.png", "changeflag": 1,
         "sentences": [{"raw": "a new road is constructed across the field"}]},
        {"split": "test", "filename": "test_000002.png", "changeflag": 0,
         "sentences": [{"raw": "the scene is almost the same as before"}]},
    ]}
    (root / "LevirCCcaptions.json").write_text(json.dumps(caps), encoding="utf-8")
    for ab in ("A", "B"):
        d = root / "images" / "test" / ab
        d.mkdir(parents=True, exist_ok=True)
        for i in range(3):
            arr = (np.random.default_rng(i).integers(0, 255, (8, 8, 3))).astype("uint8")
            Image.fromarray(arr).save(d / f"test_00000{i}.png")


@pytest.fixture
def ds(tmp_path):
    _make_fixture(tmp_path)
    return LevirCCDataset(root=tmp_path, split="test")


def test_lists_pairs(ds):
    pairs = ds.list_pairs()
    assert len(pairs) == 3
    assert all(p.t1_key == "A" and p.t2_key == "B" for p in pairs)


def test_loads_pair_images(ds):
    a, b = ds.load_pair_images(ds.list_pairs()[0])
    assert isinstance(a, Image.Image) and isinstance(b, Image.Image)


def test_caption_derived_tags(ds):
    by_loc = {p.location_id: ds.get_pair_label(p) for p in ds.list_pairs()}
    assert "building" in by_loc["test_000000"].change_type
    assert by_loc["test_000000"].stable is False
    assert "road" in by_loc["test_000001"].change_type
    assert by_loc["test_000002"].stable is True
    assert by_loc["test_000002"].change_type == "stable"


def test_split_filter(tmp_path):
    _make_fixture(tmp_path)
    with pytest.warns(UserWarning, match="no pairs found"):
        assert len(LevirCCDataset(root=tmp_path, split="train").list_pairs()) == 0
    assert len(LevirCCDataset(root=tmp_path, split="test").list_pairs()) == 3


def test_misspelled_keyword_is_named_in_a_warning(tmp_path):
    _make_fixture(tmp_path)
    with pytest.warns(UserWarning, match=r"ignoring unsupported argument.*splitt"):
        assert len(LevirCCDataset(root=tmp_path, split="test", splitt="test").list_pairs()) == 3


def test_generic_registry_options_raise_no_warning(tmp_path):
    _make_fixture(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        build_dataset("levir_cc", root=str(tmp_path), pairing="bimonthly",
                      split="test", color_mode="rgb")


def test_query_relevance(ds):
    qs = {q.text: q for q in get_queries("levir_cc")}
    assert qs, "levir_cc queries not registered"
    build_q = next(q for t, q in qs.items() if "building" in t)
    labels = {p.location_id: ds.get_pair_label(p) for p in ds.list_pairs()}
    assert build_q.predicate(labels["test_000000"]) is True
    assert build_q.predicate(labels["test_000002"]) is False
