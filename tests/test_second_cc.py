"""Unit tests for the SECOND-CC loader (captions + per-phase semantic maps), on a
tiny synthetic fixture (no network / no 2.5 GB download)."""
import json
import warnings

import numpy as np
import pytest
from PIL import Image

from src.datasets.registry import build_dataset, list_datasets
from src.datasets.second_cc import (
    CLASS_RGB,
    CLASS_TO_INDEX,
    QUERY_TO_MASK_CLASS,
    SecondCCDataset,
)
from src.queries import get_queries


def _make_fixture(root):
    caps = {"images": [
        {"split": "test", "filename": "00001_0_0.png", "changeflag": 1,
         "sentences": [{"raw": "new buildings are constructed near the road"},
                       {"raw": "many trees show up in the bareland"}]},
        {"split": "test", "filename": "00002_0_0.png", "changeflag": 0,
         "sentences": [{"raw": "nothing has changed, the scenes are identical"}]},
    ]}
    (root / "SECOND-CC-AUG.json").write_text(json.dumps(caps), encoding="utf-8")
    for ab in ("A", "B"):
        d = root / "test" / "rgb" / ab
        d.mkdir(parents=True, exist_ok=True)
        for fn in ("00001_0_0.png", "00002_0_0.png"):
            arr = np.random.default_rng(len(fn) + len(ab)).integers(0, 255, (8, 8, 3)).astype("uint8")
            Image.fromarray(arr).save(d / fn)
    # semantic maps: T1 (A) ground everywhere; T2 (B) a building block + a tree block
    sa = root / "test" / "sem" / "A"; sb = root / "test" / "sem" / "B"
    sa.mkdir(parents=True, exist_ok=True); sb.mkdir(parents=True, exist_ok=True)
    a = np.full((8, 8, 3), CLASS_RGB["ground"], dtype="uint8")
    b = a.copy()
    b[0:3, 0:3] = CLASS_RGB["building"]   # ground -> building (9 px)
    b[5:7, 5:7] = CLASS_RGB["tree"]       # ground -> tree (4 px)
    for fn in ("00001_0_0.png", "00002_0_0.png"):
        Image.fromarray(a).save(sa / fn)
    Image.fromarray(b).save(sb / "00001_0_0.png")     # changed pair
    Image.fromarray(a).save(sb / "00002_0_0.png")     # no-change pair (B == A)


@pytest.fixture
def ds(tmp_path):
    _make_fixture(tmp_path)
    return SecondCCDataset(root=tmp_path, split="test")


def test_registered():
    assert "second_cc" in list_datasets()
    assert get_queries("second_cc"), "second_cc queries not registered"


def test_lists_pairs(ds):
    pairs = ds.list_pairs()
    assert len(pairs) == 2
    assert all(p.t1_key == "A" and p.t2_key == "B" for p in pairs)


def test_caption_tags(ds):
    by = {p.location_id: ds.get_pair_label(p) for p in ds.list_pairs()}
    ct = by["00001_0_0"].change_type
    assert "building" in ct and "road" in ct and "tree" in ct and "ground" in ct
    assert by["00001_0_0"].stable is False
    assert by["00002_0_0"].stable is True
    assert by["00002_0_0"].change_type == "stable"


def test_query_relevance(ds):
    qs = {q.text: q for q in get_queries("second_cc")}
    labels = {p.location_id: ds.get_pair_label(p) for p in ds.list_pairs()}
    bq = next(q for t, q in qs.items() if "building" in t)
    assert bq.predicate(labels["00001_0_0"]) is True
    assert bq.predicate(labels["00002_0_0"]) is False


def test_class_change_mask(ds):
    p = next(p for p in ds.list_pairs() if p.location_id == "00001_0_0")
    bm = ds.load_change_mask(p, "building")
    tm = ds.load_change_mask(p, "tree")
    assert bm.dtype == bool and bm.sum() == 9
    assert tm.sum() == 4
    assert not (bm & tm).any()
    # change_class=None -> uint8 class-index map of the change (T2 class where changed,
    # 0 elsewhere), matching LevirMCIDataset's contract.
    idx = ds.load_change_mask(p)
    assert idx.dtype == np.uint8
    assert (idx > 0).sum() == 13                              # 9 building + 4 tree changed
    assert (idx == CLASS_TO_INDEX["building"]).sum() == 9
    assert (idx == CLASS_TO_INDEX["tree"]).sum() == 4


def test_transition_mask(ds):
    p = next(p for p in ds.list_pairs() if p.location_id == "00001_0_0")
    assert ds.transition_change_mask(p, "ground", "building").sum() == 9
    assert ds.transition_change_mask(p, "ground", "tree").sum() == 4
    assert ds.transition_change_mask(p, "water", "building").sum() == 0


def test_no_change_pair_empty_mask(ds):
    p = next(p for p in ds.list_pairs() if p.location_id == "00002_0_0")
    assert not ds.load_change_mask(p).any()


def test_query_to_mask_classes_valid():
    assert set(QUERY_TO_MASK_CLASS.values()).issubset(set(CLASS_TO_INDEX))


def test_build_via_registry(tmp_path):
    _make_fixture(tmp_path)
    d = build_dataset("second_cc", root=str(tmp_path), split="test")
    assert d.name == "second_cc" and len(d.list_pairs()) == 2


def test_misspelled_keyword_is_named_in_a_warning(tmp_path):
    _make_fixture(tmp_path)
    with pytest.warns(UserWarning, match=r"ignoring unsupported argument.*splitt"):
        assert len(SecondCCDataset(root=tmp_path, split="test", splitt="test").list_pairs()) == 2


def test_generic_registry_options_raise_no_warning(tmp_path):
    _make_fixture(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        build_dataset("second_cc", root=str(tmp_path), pairing="bimonthly",
                      split="test", color_mode="rgb")
