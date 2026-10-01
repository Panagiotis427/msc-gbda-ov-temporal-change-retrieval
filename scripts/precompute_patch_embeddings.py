"""
Offline warm of the per-patch embedding cache -> instant first patch query.

Localised patch-level Δ-similarity (report §8.1, the headline DEN config) needs
per-patch embeddings for the *whole* corpus. Computing them at the first
``approach="patch"`` query stalls the Gradio app for a full GPU pass over every
pair. This script precomputes + caches them so the app loads them instantly on
startup (the instant-search precompute backend).

The patch cache is keyed by ``(dataset, encoder, split, color_mode, lora)`` via
the single source of truth in ``src.embeddings.cache_tag_for`` and its rows are
aligned to the *pair* store's pair list, so the app can index them positionally.
Idempotent: a matching cache is reused unless ``--force``.

CLI (mirrors ``python -m src.embeddings``):
    python -m scripts.precompute_patch_embeddings --dataset dynamic_earthnet \
        --root data/DynamicEarthNet --encoder clip_vitl14 --split test --color-mode nrg
"""
from __future__ import annotations

import argparse

from src.datasets.registry import build_dataset
from src.embeddings import cache_tag_for, load_or_compute, load_or_compute_patches
from src.encoders import get_encoder


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Precompute the per-patch embedding cache (instant patch search).")
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
    ap.add_argument("--color-mode", default="rgb", choices=["rgb", "nrg", "ndvi"])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    ds = build_dataset(
        args.dataset, root=args.root, pairing=args.pairing,
        split=None if args.split == "all" else args.split,
        color_mode=args.color_mode,
    )
    enc = get_encoder(args.encoder)
    # No lora component: the app's patch path encodes with the plain encoder (LoRA is never merged
    # into it), so the patch cache is encoder-LoRA-agnostic — tagging it "_lora" would mislabel it.
    cache_tag = cache_tag_for(args.split, args.color_mode)

    # The pair store fixes the canonical, load-failure-pruned pair order; patch
    # rows must align to it so the app can index them by store position.
    pair_store = load_or_compute(ds, enc, cache_dir=args.cache_dir, cache_tag=cache_tag)
    patch_store = load_or_compute_patches(
        ds, enc, pair_store.pairs, cache_dir=args.cache_dir, force=args.force,
        batch_size=args.batch_size, cache_tag=cache_tag, progress=True,
    )
    print(f"dataset={patch_store.dataset_name} encoder={patch_store.encoder_name} "
          f"N={len(patch_store)} patch_shape={patch_store.patch_t1.shape}")


if __name__ == "__main__":
    main()
