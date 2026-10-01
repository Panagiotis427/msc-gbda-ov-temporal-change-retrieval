"""
Fast tests for the per-patch embedding cache (``src.embeddings`` patch store).

Pure-numpy, CPU-only — no CLIP, no network, no DEN fixture. A tiny fake dataset
+ a deterministic patch encoder exercise:
  - save/load round-trip of ``PatchEmbeddingStore`` (arrays + pair order),
  - ``load_or_compute_patches`` reuses a matching cache,
  - it recomputes (not silently reuses) when the on-disk pair set is stale,
    mirroring the order-sensitive guard of the pair store.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.datasets.base import PairKey
from src.embeddings import (
    PatchEmbeddingStore,
    compute_patch_embeddings,
    load_or_compute_patches,
    patch_cache_path,
)


class _FakeDataset:
    """Deterministic stand-in: each pair maps to two fixed-size grey images whose
    intensity encodes the pair index, so patch embeddings differ per pair."""

    name = "fake_ds"

    def __init__(self, n: int = 4) -> None:
        self._pairs = [PairKey(f"loc{i}", "t1", "t2") for i in range(n)]

    def list_pairs(self):
        return list(self._pairs)

    def load_pair_images(self, pair: PairKey):
        from PIL import Image
        idx = int(pair.location_id.removeprefix("loc"))
        a = Image.new("RGB", (8, 8), color=(idx, 0, 0))
        b = Image.new("RGB", (8, 8), color=(0, idx, 0))
        return a, b


class _PatchEncoder:
    """Returns deterministic per-patch embeddings: shape [B, P=4, D=3] keyed by
    each image's mean colour, so rows stay distinguishable per pair."""

    name = "fake_patch_enc"

    def encode_image_patches(self, images):
        out = []
        for im in images:
            m = np.array(im, dtype=np.float32).reshape(-1, 3).mean(0)
            out.append(np.broadcast_to(m, (4, 3)).copy())
        return np.stack(out)  # [B, 4, 3]


def test_patch_store_roundtrip(tmp_path):
    ds, enc = _FakeDataset(4), _PatchEncoder()
    store = compute_patch_embeddings(ds, enc, ds.list_pairs())
    assert store.patch_t1.shape == (4, 4, 3)
    path = patch_cache_path(tmp_path, ds.name, enc.name, tag="test")
    store.save(path)

    loaded = PatchEmbeddingStore.load(path)
    assert [tuple(p) for p in loaded.pairs] == [tuple(p) for p in store.pairs]
    np.testing.assert_array_equal(loaded.patch_t1, store.patch_t1)
    np.testing.assert_array_equal(loaded.patch_t2, store.patch_t2)


def test_load_or_compute_patches_reuses_valid_cache(tmp_path):
    ds, enc = _FakeDataset(4), _PatchEncoder()
    pairs = ds.list_pairs()
    first = load_or_compute_patches(ds, enc, pairs, cache_dir=tmp_path, cache_tag="test")
    # Corrupt the on-disk arrays in place; a reuse must return THESE values
    # (proving it read the cache rather than recomputing from the encoder).
    path = patch_cache_path(tmp_path, ds.name, enc.name, tag="test")
    sentinel = PatchEmbeddingStore(
        dataset_name=ds.name, encoder_name=enc.name, pairs=first.pairs,
        patch_t1=np.full_like(first.patch_t1, 7.0),
        patch_t2=np.full_like(first.patch_t2, 9.0),
    )
    sentinel.save(path)

    again = load_or_compute_patches(ds, enc, pairs, cache_dir=tmp_path, cache_tag="test")
    assert np.allclose(again.patch_t1, 7.0) and np.allclose(again.patch_t2, 9.0)


def test_load_or_compute_patches_recomputes_on_stale_pairset(tmp_path):
    ds, enc = _FakeDataset(4), _PatchEncoder()
    full = compute_patch_embeddings(ds, enc, ds.list_pairs())
    # Persist a stale cache missing the last pair -> pair set differs from the
    # requested pairs -> must recompute over the full set, not reuse.
    stale = PatchEmbeddingStore(
        dataset_name=ds.name, encoder_name=enc.name, pairs=full.pairs[:-1],
        patch_t1=full.patch_t1[:-1], patch_t2=full.patch_t2[:-1],
    )
    stale.save(patch_cache_path(tmp_path, ds.name, enc.name, tag="test"))

    got = load_or_compute_patches(ds, enc, ds.list_pairs(), cache_dir=tmp_path, cache_tag="test")
    assert len(got) == 4
    assert [tuple(p) for p in got.pairs] == [tuple(p) for p in ds.list_pairs()]


def test_interrupted_save_keeps_the_previous_cache(tmp_path, monkeypatch):
    ds, enc = _FakeDataset(4), _PatchEncoder()
    store = compute_patch_embeddings(ds, enc, ds.list_pairs())
    path = patch_cache_path(tmp_path, ds.name, enc.name, tag="test")
    store.save(path)
    before = path.read_bytes()

    def interrupted(fh, **arrays):
        fh.write(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr(np, "savez", interrupted)
    with pytest.raises(OSError):
        store.save(path)
    monkeypatch.undo()

    assert path.read_bytes() == before                    # the earlier cache is intact
    assert [p.name for p in tmp_path.iterdir()] == [path.name]   # no temp file left behind
    assert len(PatchEmbeddingStore.load(path)) == 4


def test_save_names_the_file_as_np_savez_does(tmp_path):
    ds, enc = _FakeDataset(2), _PatchEncoder()
    store = compute_patch_embeddings(ds, enc, ds.list_pairs())
    store.save(tmp_path / "plain_name")                   # np.savez appends ".npz"
    assert [p.name for p in tmp_path.iterdir()] == ["plain_name.npz"]
    assert len(PatchEmbeddingStore.load(tmp_path / "plain_name.npz")) == 2
