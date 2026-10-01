"""
Per-pair embedding computation + on-disk cache.

For every bi-temporal pair ``(T1, T2)`` of a ``TemporalDataset`` this module
runs the frozen image encoder on both timesteps and stores the resulting
L2-normalised vectors ``f_T1, f_T2`` (shape ``[N, D]`` each, aligned with an
ordered pair list). This is the artefact every retrieval/benchmark/training
step consumes — it decouples the (slow, GPU) encoding pass from the (fast,
CPU) scoring passes and makes runs reproducible.

Cache file: ``<cache_dir>/<dataset>__<encoder>[__<tag>]__pair_embeddings.npz``
where ``<tag> = <split>[_<color>][_lora]`` (see ``cache_tag_for``); the rgb
default adds no colour suffix.

CLI:
    python -m src.embeddings --dataset dynamic_earthnet \
        --root data/DynamicEarthNet --encoder clip_vitl14
"""
from __future__ import annotations

import argparse
import contextlib
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import torch

from src.datasets.base import PairKey, TemporalDataset
from src.encoders import get_encoder
from src.features import compute_change_feature


def _savez_atomic(path: str | Path, **arrays) -> None:
    """``np.savez`` into a temporary file beside *path*, then ``os.replace`` it into
    place: an interrupted write leaves the previous cache (or none) rather than a
    truncated .npz the next run cannot load. The file name is what ``np.savez``
    would have used (it appends ``.npz`` to any other name)."""
    path = Path(path)
    if not str(path).endswith(".npz"):
        path = path.with_name(path.name + ".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        with open(tmp, "wb") as fh:
            np.savez(fh, **arrays)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise


def cache_path(cache_dir: str | Path, dataset_name: str, encoder_name: str,
               tag: str = "") -> Path:
    suffix = f"__{tag}" if tag else ""
    return Path(cache_dir) / f"{dataset_name}__{encoder_name}{suffix}__pair_embeddings.npz"


def patch_cache_path(cache_dir: str | Path, dataset_name: str, encoder_name: str,
                     tag: str = "") -> Path:
    """Path of the per-patch embedding cache (sibling of :func:`cache_path`),
    keyed by the same ``<split>[_<color>][_lora]`` tag so it never aliases another
    split/colour combination."""
    suffix = f"__{tag}" if tag else ""
    return Path(cache_dir) / f"{dataset_name}__{encoder_name}{suffix}{patch_version_suffix(encoder_name)}__patch_embeddings.npz"


def patch_version_suffix(encoder_name: str) -> str:
    """``__ctx`` for the open_clip encoders, whose per-patch tokens changed at a07b8a6 (before it, each patch
    passed through the transformer without attending to the others), so an older cache is never reused; the
    Hugging Face CLIP encoder was always contextual and keeps its cache names."""
    return "__ctx" if encoder_name in ("georsclip", "remoteclip") else ""


_KNOWN_COLORS = ("rgb", "nrg", "ndvi")


def color_tag(color_mode: str = "rgb") -> str:
    """Canonical colour suffix for cache/artefact names: ``_<color>`` (empty for
    the rgb default). Single source of truth shared by ``cache_tag_for`` and the
    figure/export scripts so the suffix can never drift between them."""
    return f"_{color_mode}" if color_mode != "rgb" else ""


def cache_tag_for(split: str, color_mode: str = "rgb", lora: bool = False) -> str:
    """Canonical embedding-cache tag: ``<split>[_<color>][_lora]``.

    Single source of truth for the cache-tag string, matching the layout the
    committed caches were written with by ``scripts.run_pipeline`` (rgb adds no
    colour suffix; lora appends ``_lora``). Importers: ``run_pipeline``,
    ``train``, ``cv_eval``, ``eval_rerank``, ``make_comparison_figure``, the
    ``embeddings`` CLI, ``app``, ``export_results`` (grep ``cache_tag_for`` for
    the authoritative list). Keeping this in one place avoids the historical
    ``test``+``rgb``->empty-tag drift.

    The tag is positional (no delimiter between fields), so a ``split`` whose name
    ended in ``_<color>`` or ``_lora`` could alias another (split, color, lora)
    combination — e.g. ``("test_nrg", "rgb")`` and ``("test", "nrg")`` both →
    ``"test_nrg"``. Splits are a closed set ({train,val,test,all}) so this never
    happens in practice; we assert it to keep the collision impossible rather than
    merely improbable.
    """
    if color_mode not in _KNOWN_COLORS:
        raise ValueError(f"unknown color_mode {color_mode!r}; expected one of {_KNOWN_COLORS}")
    if any(split.endswith(f"_{c}") for c in _KNOWN_COLORS) or split.endswith("_lora"):
        raise ValueError(
            f"split {split!r} ends with a colour/lora suffix and would alias another "
            "cache tag; rename the split."
        )
    lora_tag = "_lora" if lora else ""
    return f"{split}{color_tag(color_mode)}{lora_tag}"


@dataclass
class PairEmbeddingStore:
    """Ordered pair list + aligned ``f_T1`` / ``f_T2`` matrices."""

    dataset_name: str
    encoder_name: str
    embed_dim: int
    pairs: List[PairKey]
    f_t1: np.ndarray  # [N, D] float32, L2-normalised
    f_t2: np.ndarray  # [N, D] float32, L2-normalised

    def __len__(self) -> int:
        return len(self.pairs)

    def change_features(self, mode: str = "difference") -> np.ndarray:
        """Δf for every pair via :func:`src.features.compute_change_feature`."""
        t1 = torch.from_numpy(self.f_t1)
        t2 = torch.from_numpy(self.f_t2)
        return compute_change_feature(t1, t2, mode=mode).numpy().astype(np.float32)

    def save(self, path: str | Path) -> None:
        _savez_atomic(
            path,
            f_t1=self.f_t1.astype(np.float32),
            f_t2=self.f_t2.astype(np.float32),
            loc=np.array([p.location_id for p in self.pairs]),
            t1=np.array([p.t1_key for p in self.pairs]),
            t2=np.array([p.t2_key for p in self.pairs]),
            dataset_name=np.array(self.dataset_name),
            encoder_name=np.array(self.encoder_name),
            embed_dim=np.array(self.embed_dim),
        )

    @classmethod
    def load(cls, path: str | Path) -> "PairEmbeddingStore":
        with np.load(path, allow_pickle=False) as d:
            pairs = [
                PairKey(str(l), str(a), str(b))
                for l, a, b in zip(d["loc"], d["t1"], d["t2"])
            ]
            return cls(
                dataset_name=str(d["dataset_name"]),
                encoder_name=str(d["encoder_name"]),
                embed_dim=int(d["embed_dim"]),
                pairs=pairs,
                f_t1=d["f_t1"].astype(np.float32),
                f_t2=d["f_t2"].astype(np.float32),
            )


def compute_pair_embeddings(
    dataset: TemporalDataset,
    encoder,
    batch_size: int = 32,
) -> PairEmbeddingStore:
    """Encode both timesteps of every pair. Pairs whose tiles fail to load are
    skipped (real DEN occasionally misses a monthly tile)."""
    pairs: List[PairKey] = []
    imgs_t1, imgs_t2 = [], []
    for pair in dataset.list_pairs():
        try:
            a, b = dataset.load_pair_images(pair)
        except FileNotFoundError as exc:
            print(f"  skip {pair}: {exc}")
            continue
        pairs.append(pair)
        imgs_t1.append(a)
        imgs_t2.append(b)

    if not pairs:
        raise RuntimeError("No loadable pairs in dataset.")

    print(f"Encoding {len(pairs)} pairs ({2 * len(pairs)} images) with "
          f"'{encoder.name}' on {encoder.device} ...")
    f_t1 = encoder.encode_image(imgs_t1, batch_size=batch_size).astype(np.float32)
    f_t2 = encoder.encode_image(imgs_t2, batch_size=batch_size).astype(np.float32)

    return PairEmbeddingStore(
        dataset_name=dataset.name,
        encoder_name=encoder.name,
        embed_dim=int(f_t1.shape[1]),
        pairs=pairs,
        f_t1=f_t1,
        f_t2=f_t2,
    )


def load_or_compute(
    dataset: TemporalDataset,
    encoder,
    cache_dir: str | Path = "data/cache",
    force: bool = False,
    batch_size: int = 32,
    cache_tag: str = "",
) -> PairEmbeddingStore:
    path = cache_path(cache_dir, dataset.name, encoder.name, tag=cache_tag)
    if path.exists() and not force:
        store = PairEmbeddingStore.load(path)
        expected = [tuple(p) for p in dataset.list_pairs()]
        # Order-sensitive list equality is intentional: f_t1/f_t2 rows are aligned
        # positionally to this pair list, so a reordered-but-equal pair set is NOT
        # safely reusable and must be recomputed (the worst case here is a redundant
        # recompute, never silent row/label misalignment).
        if [tuple(p) for p in store.pairs] == expected:
            print(f"Loaded {len(store)} pair embeddings (mode=reused) from cache: {path}")
            return store
        print(f"Cache {path} stale (pair set changed: "
              f"{len(store.pairs)} cached vs {len(expected)} expected) "
              "-- recomputing.")
    store = compute_pair_embeddings(dataset, encoder, batch_size=batch_size)
    store.save(path)
    print(f"Saved {len(store)} pair embeddings (mode=recomputed) -> {path}")
    return store


# ---------------------------------------------------------------------------
# Per-patch embedding cache (localised / patch-level retrieval, report §8.1)
# ---------------------------------------------------------------------------
# Patch-level Δ-similarity needs per-patch embeddings for the *whole* corpus.
# Encoding them at first ``approach="patch"`` query stalls the UI for the length
# of a full GPU pass over every pair. This store decouples that encode (slow,
# GPU) from the scoring pass (fast, CPU) exactly as the pair store does — a warm
# cache makes the first patch query instant (the instant-search precompute).
# Rows are aligned positionally to an ordered pair list; the pair list is the
# pair store's (so engine code can index ``patch_t1[i]`` with the same ``i`` it
# uses on ``store.pairs[i]``).


@dataclass
class PatchEmbeddingStore:
    """Ordered pair list + aligned per-patch ``patch_t1`` / ``patch_t2`` tensors
    (shape ``[N, P, D]`` each)."""

    dataset_name: str
    encoder_name: str
    pairs: List[PairKey]
    patch_t1: np.ndarray  # [N, P, D] float32
    patch_t2: np.ndarray  # [N, P, D] float32

    def __len__(self) -> int:
        return len(self.pairs)

    def save(self, path: str | Path) -> None:
        _savez_atomic(
            path,
            patch_t1=self.patch_t1.astype(np.float32),
            patch_t2=self.patch_t2.astype(np.float32),
            loc=np.array([p.location_id for p in self.pairs]),
            t1=np.array([p.t1_key for p in self.pairs]),
            t2=np.array([p.t2_key for p in self.pairs]),
            dataset_name=np.array(self.dataset_name),
            encoder_name=np.array(self.encoder_name),
        )

    @classmethod
    def load(cls, path: str | Path) -> "PatchEmbeddingStore":
        with np.load(path, allow_pickle=False) as d:
            pairs = [
                PairKey(str(l), str(a), str(b))
                for l, a, b in zip(d["loc"], d["t1"], d["t2"])
            ]
            return cls(
                dataset_name=str(d["dataset_name"]),
                encoder_name=str(d["encoder_name"]),
                pairs=pairs,
                patch_t1=d["patch_t1"].astype(np.float32),
                patch_t2=d["patch_t2"].astype(np.float32),
            )


def compute_patch_embeddings(
    dataset: TemporalDataset,
    encoder,
    pairs: List[PairKey],
    batch_size: int = 32,
    progress: bool = False,
) -> PatchEmbeddingStore:
    """Encode per-patch embeddings for every pair in *pairs* (kept in the given
    order so the result aligns with the pair store). Chunks the GPU passes to
    bound host memory. Raises if the encoder exposes no patch tokens."""
    pairs = list(pairs)
    if not pairs:
        raise RuntimeError("No pairs to encode patches for.")
    p1, p2 = [], []
    chunks = range(0, len(pairs), batch_size)
    if progress:
        try:
            from tqdm import tqdm
            chunks = tqdm(chunks, desc=f"patch-encode {encoder.name}", unit="batch")
        except ImportError:
            pass
    for s in chunks:
        imgs1, imgs2 = [], []
        for pk in pairs[s:s + batch_size]:
            try:
                im1, im2 = dataset.load_pair_images(pk)
            except FileNotFoundError as exc:
                # Do NOT skip (rows must stay positionally aligned to the pair store); fail with a
                # clear, actionable message instead of an opaque crash deep in the encoder loop.
                raise RuntimeError(
                    f"patch encode: tile for pair {pk} is missing ({exc}); the pair set changed since "
                    "the pair-embedding cache was built — delete the matching *__patch_embeddings.npz "
                    "(and *__pair_embeddings.npz) and recompute.") from exc
            imgs1.append(im1)
            imgs2.append(im2)
        a = encoder.encode_image_patches(imgs1)
        b = encoder.encode_image_patches(imgs2)
        if a is None or b is None:
            raise RuntimeError(
                f"{encoder.name} exposes no patch tokens; pick zero_shot/naive/peft.")
        p1.append(np.asarray(a))
        p2.append(np.asarray(b))
    return PatchEmbeddingStore(
        dataset_name=dataset.name,
        encoder_name=encoder.name,
        pairs=pairs,
        patch_t1=np.concatenate(p1, axis=0).astype(np.float32),
        patch_t2=np.concatenate(p2, axis=0).astype(np.float32),
    )


def load_or_compute_patches(
    dataset: TemporalDataset,
    encoder,
    pairs: List[PairKey],
    cache_dir: str | Path = "data/cache",
    force: bool = False,
    batch_size: int = 32,
    cache_tag: str = "",
    progress: bool = False,
) -> PatchEmbeddingStore:
    """Reuse the on-disk patch cache when its pair list matches *pairs*, else
    recompute + save. Mirrors :func:`load_or_compute`'s order-sensitive guard:
    a reordered-but-equal pair set is recomputed rather than risk row/label
    misalignment."""
    path = patch_cache_path(cache_dir, dataset.name, encoder.name, tag=cache_tag)
    expected = [tuple(p) for p in pairs]
    if path.exists() and not force:
        store = PatchEmbeddingStore.load(path)
        if [tuple(p) for p in store.pairs] == expected:
            print(f"Loaded {len(store)} patch embeddings (mode=reused) from cache: {path}")
            return store
        print(f"Patch cache {path} stale (pair set changed: "
              f"{len(store.pairs)} cached vs {len(expected)} expected) -- recomputing.")
    store = compute_patch_embeddings(dataset, encoder, pairs,
                                     batch_size=batch_size, progress=progress)
    store.save(path)
    print(f"Saved {len(store)} patch embeddings (mode=recomputed) -> {path}")
    return store


def main() -> None:
    ap = argparse.ArgumentParser(description="Precompute per-pair embeddings cache")
    ap.add_argument("--dataset", default="dynamic_earthnet")
    ap.add_argument("--root", default="data/DynamicEarthNet",
                    help="Dataset root (DEN) or ignored for cache-only datasets")
    ap.add_argument("--pairing", default="bimonthly",
                    choices=["bimonthly", "monthly", "seasonal-quartet"])
    ap.add_argument("--encoder", default="clip_vitl14")
    ap.add_argument("--cache-dir", default="data/cache")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--split", default="test",
                    help="DEN preprocessed split: train|val|test|all")
    ap.add_argument("--color-mode", default="rgb",
                    choices=["rgb", "nrg", "ndvi"])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    from src.datasets.registry import build_dataset
    color_mode = args.color_mode
    ds = build_dataset(
        args.dataset, root=args.root, pairing=args.pairing,
        split=None if args.split == "all" else args.split,
        color_mode=color_mode,
    )
    enc = get_encoder(args.encoder)
    cache_tag = cache_tag_for(args.split, color_mode)
    store = load_or_compute(
        ds, enc, cache_dir=args.cache_dir, force=args.force,
        batch_size=args.batch_size, cache_tag=cache_tag,
    )
    print(f"dataset={store.dataset_name} encoder={store.encoder_name} "
          f"N={len(store)} D={store.embed_dim}")


if __name__ == "__main__":
    main()
