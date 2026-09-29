"""QFabric status-transition loader + query set — synthetic fixture, no net/GPU."""
from __future__ import annotations

import json

import numpy as np
from PIL import Image

from src.datasets.base import PairKey, PairLabel
from src.datasets.registry import build_dataset, list_datasets


def _write_fixture(tmp_path):
    """Two crops. 100: greenland->land_cleared->construction_done (3 tp).
    200: prior_construction->construction_done (transition, 2 tp)."""
    imgs = tmp_path / "QFabric"
    imgs.mkdir()
    status = {
        "100_0_256": {"d1": "greenland", "d2": "land_cleared", "d3": "construction_done"},
        "200_0_256": {"d1": "prior_construction", "d2": "construction_done"},
    }
    for ck, days in status.items():
        loc, xoff, yoff = ck.split("_")
        for i in range(1, len(days) + 1):
            Image.new("RGB", (16, 16), (i * 40, 60, 70)).save(
                imgs / f"{loc}.d{i}.0101201{i}_{xoff}_{yoff}.tif")
    labels = tmp_path / "qfabric_status_labels.json"
    labels.write_text(json.dumps(status))
    return str(imgs), str(labels), status


def test_registered():
    assert "qfabric_status" in list_datasets()


def test_pairs_and_transition_labels(tmp_path):
    root, labels, _ = _write_fixture(tmp_path)
    ds = build_dataset("qfabric_status", root=root, labels_path=labels,
                       color_mode="rgb", pairing="bimonthly", split="all")  # generic dropped
    assert ds.name == "qfabric_status"
    pairs = ds.list_pairs()
    assert len(pairs) == 3  # crop100: 2 pairs, crop200: 1 pair
    by_loc = {p.location_id: p for p in pairs}
    # crop 200's single pair is a real transition prior_construction -> construction_done
    lb = ds.get_pair_label(by_loc["200_0_256"])
    assert lb.dominant_t1_class == "prior_construction"
    assert lb.dominant_t2_class == "construction_done"
    assert lb.change_type == "prior_construction->construction_done"
    assert lb.stable is False


def test_stable_pair_flagged(tmp_path):
    root, labels, _ = _write_fixture(tmp_path)
    ds = build_dataset("qfabric_status", root=root, labels_path=labels)
    # construct a stable PairKey directly (same status both ends)
    # crop 100 d3 is construction_done; fake a d3->d3 to check stable detection
    lb = PairLabel  # noqa: F841  (import sanity)
    pk = PairKey(location_id="100_0_256", t1_key="d3", t2_key="d3")
    got = ds.get_pair_label(pk)
    assert got.stable is True and got.dominant_t1_class == got.dominant_t2_class


def test_unlabelled_endpoint_returns_none(tmp_path):
    root, labels, _ = _write_fixture(tmp_path)
    ds = build_dataset("qfabric_status", root=root, labels_path=labels)
    # d9 has no status -> label is None (not a crash, not a default)
    pk = PairKey(location_id="100_0_256", t1_key="d1", t2_key="d9")
    assert ds.get_pair_label(pk) is None


def test_caption_is_phrase_not_token(tmp_path):
    root, labels, _ = _write_fixture(tmp_path)
    ds = build_dataset("qfabric_status", root=root, labels_path=labels)
    for p in ds.list_pairs():
        cap = ds.text_caption_for_pair(p)
        assert isinstance(cap, str) and " " in cap  # templated phrase


def test_queries_registered_and_transition_discriminative():
    from src.queries import get_queries
    qs = get_queries("qfabric_status")
    assert len(qs) >= 5
    completed = next(q for q in qs if "completed" in q.text)
    # relevant: a real transition INTO a finished state
    trans = PairLabel(change_type="land_cleared->construction_done", stable=False,
                      dominant_t1_class="land_cleared", dominant_t2_class="construction_done")
    assert completed.predicate(trans) is True
    # NOT relevant: stable "already done" pair (same status both ends) — the
    # hard negative that separates zero_shot Δ from naive after-image content
    stable = PairLabel(change_type="construction_done->construction_done", stable=True,
                       dominant_t1_class="construction_done", dominant_t2_class="construction_done")
    assert completed.predicate(stable) is False


def test_split_disjoint(tmp_path):
    """5 crops ending construction_done + 5 ending land_cleared; crop-level split."""
    imgs = tmp_path / "QFabric"; imgs.mkdir()
    status = {}
    for ci, final in enumerate(["construction_done", "land_cleared"]):
        for k in range(5):
            ck = f"{ci}{k}_0_0"
            loc, xoff, yoff = ck.split("_")
            for i in (1, 2):
                Image.new("RGB", (8, 8), (k * 20, ci * 30, 40)).save(
                    imgs / f"{loc}.d{i}.0101201{i}_{xoff}_{yoff}.tif")
            status[ck] = {"d1": "greenland", "d2": final}
    labels = tmp_path / "qfabric_status_labels.json"
    labels.write_text(json.dumps(status))
    tr = set(build_dataset("qfabric_status", root=str(imgs), labels_path=str(labels),
                           split="train", train_frac=0.8, seed=42).list_locations())
    te = set(build_dataset("qfabric_status", root=str(imgs), labels_path=str(labels),
                           split="test", train_frac=0.8, seed=42).list_locations())
    assert tr and te and tr.isdisjoint(te)
    assert len(tr) == 8 and len(te) == 2   # 80/20 stratified by final status


def test_end_to_end_benchmark(tmp_path):
    from src.embeddings import compute_pair_embeddings
    from src.retrieval import ChangeRetriever
    from src.benchmark import run_benchmark

    root, labels, _ = _write_fixture(tmp_path)
    ds = build_dataset("qfabric_status", root=root, labels_path=labels)

    class _Enc:
        name = "mock"; embed_dim = 4
        import torch as _t
        device = _t.device("cpu")
        def encode_image(self, images, batch_size=32):
            return np.ones((len(images), 4), dtype=np.float32)
        def encode_text(self, texts, batch_size=32):
            t = [texts] if isinstance(texts, str) else texts
            return np.ones((len(t), 4), dtype=np.float32)

    store = compute_pair_embeddings(ds, _Enc())
    rep = run_benchmark(ds, ChangeRetriever(store, _Enc()), approach="zero_shot")
    assert 0.0 <= rep.mAP <= 1.0
