"""
Regenerate benchmark results from CACHED embeddings + adapters — no encoding,
no training. Verifies the headline mAP table and populates ``results/``
(one JSON per run + a flat ``macro_summary.csv``) for the figure / error-analysis
scripts to consume.

For each (encoder x split x color x approach) it loads the cached
:class:`PairEmbeddingStore` and, for ``peft``, the cached ``ProjectionHead``
adapter, runs the label-grounded benchmark, and writes the JSON. Combos whose
cache or adapter is missing are **skipped with a warning** — never silently,
and never recomputed (we load the .npz directly rather than via
``load_or_compute`` so a cache miss can't trigger a heavy re-encode).

Examples
--------
Headline DEN table (all encoders, all splits, rgb, all approaches)::

    python -m scripts.export_results --results-dir results

Single combo (fast check)::

    python -m scripts.export_results --encoders clip_vitl14 --eval-splits train \\
        --approaches naive zero_shot peft --results-dir results

Colour ablation + LoRA::

    python -m scripts.export_results --color-modes rgb nrg ndvi --lora --results-dir results
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import json

from src.benchmark import run_benchmark
from src.datasets.registry import build_dataset
from src.embeddings import PairEmbeddingStore, cache_path, cache_tag_for
from src.encoders import get_encoder
from src.error_analysis import build_confusion
from src.model import adapter_path, load_adapter
from src.results_io import load_all, result_path, write_macro_csv, write_report
from src.retrieval import ChangeRetriever


def export_one(
    dataset: str,
    encoder_name: str,
    split: str,
    color: str,
    approach: str,
    *,
    enc,
    ds,
    cache_dir: str,
    results_dir: str,
    lora: bool = False,
    confusion: bool = False,
    train_split: str = "train",
    mode: str = "difference",
) -> Optional[dict]:
    """Benchmark one combo from cache and write its JSON. Returns the record
    dict (for the macro CSV) or ``None`` if skipped."""
    tag = cache_tag_for(split, color, lora=lora)
    cpath = cache_path(cache_dir, dataset, encoder_name, tag=tag)
    label = f"{encoder_name}/{split}/{color}/{approach}{' +lora' if lora else ''}"
    if not cpath.exists():
        print(f"  skip {label}: no embedding cache ({cpath.name})")
        return None

    store = PairEmbeddingStore.load(cpath)
    if enc.embed_dim != store.embed_dim:
        print(f"  skip {label}: encoder dim {enc.embed_dim} != store {store.embed_dim}")
        return None
    retr = ChangeRetriever(store, enc)

    if approach == "peft":
        apath = adapter_path(dataset, encoder_name, color,
                             train_split=train_split, mode=mode)
        if not apath.exists():
            print(f"  skip {label}: no adapter ({apath.name})")
            return None
        adapter, meta = load_adapter(str(apath))
        retr.set_adapter(adapter, feature_mode=meta.get("feature_mode", "difference"))

    report = run_benchmark(ds, retr, approach=approach)
    p = result_path(results_dir, dataset, encoder_name, split,
                    color=color, approach=approach, lora=lora)
    write_report(report, p, color_mode=color, split=split, lora=lora)
    print(f"  wrote {p.name:64s} mAP={report.mAP:.4f}  N={report.n_pairs}")

    if confusion:
        conf = build_confusion(ds, retr, approach=approach, split=split)
        lora_tag = "__lora" if lora else ""
        conf_dir = Path(results_dir) / "confusion"
        conf_dir.mkdir(parents=True, exist_ok=True)
        conf_path = conf_dir / (
            f"{dataset}__{encoder_name}__{split}__{color}__{approach}{lora_tag}"
            "__confusion.json"
        )
        with open(conf_path, "w", encoding="utf-8") as f:
            json.dump(conf.to_dict(), f, indent=2, sort_keys=True)
            f.write("\n")
        print(f"        confusion -> {conf_path.name}")

    return report.to_dict(color_mode=color, split=split, lora=lora)


def main() -> None:
    ap = argparse.ArgumentParser(description="Export cached benchmarks to JSON/CSV")
    ap.add_argument("--dataset", default="dynamic_earthnet")
    ap.add_argument("--root", default="data/DynamicEarthNet")
    ap.add_argument("--pairing", default="bimonthly",
                    choices=["bimonthly", "monthly", "seasonal-quartet"])
    ap.add_argument("--encoders", nargs="+",
                    default=["clip_vitl14", "georsclip", "remoteclip"])
    ap.add_argument("--eval-splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--color-modes", nargs="+", default=["rgb"])
    ap.add_argument("--approaches", nargs="+",
                    default=["naive", "zero_shot", "peft"])
    ap.add_argument("--train-split", default="train",
                    help="Adapter training split to locate for peft export "
                         "(matches run_pipeline --train-split; default: train).")
    ap.add_argument("--mode", default="difference",
                    choices=["difference", "concatenate"],
                    help="Adapter feature mode to locate for peft export "
                         "(matches run_pipeline --mode; default: difference).")
    ap.add_argument("--lora", action="store_true",
                    help="Also export zero_shot on LoRA-merged embeddings where cached.")
    ap.add_argument("--confusion", action="store_true",
                    help="Also write a per-query confusion report to results/confusion/.")
    ap.add_argument("--cache-dir", default="data/cache")
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args()

    written: List[dict] = []
    for enc_name in args.encoders:
        print(f"\n=== encoder: {enc_name} ===")
        enc = get_encoder(enc_name)
        for split in args.eval_splits:
            # Labels are colour-independent -> build the dataset once per split.
            ds = build_dataset(args.dataset, root=args.root,
                               pairing=args.pairing, split=split)
            for color in args.color_modes:
                for approach in args.approaches:
                    rec = export_one(
                        args.dataset, enc_name, split, color, approach,
                        enc=enc, ds=ds, cache_dir=args.cache_dir,
                        results_dir=args.results_dir, confusion=args.confusion,
                        train_split=args.train_split, mode=args.mode,
                    )
                    if rec:
                        written.append(rec)
                if args.lora:
                    rec = export_one(
                        args.dataset, enc_name, split, color, "zero_shot",
                        enc=enc, ds=ds, cache_dir=args.cache_dir,
                        results_dir=args.results_dir, lora=True,
                        confusion=args.confusion,
                    )
                    if rec:
                        written.append(rec)

    if written:
        # Rebuild the macro CSV from ALL results on disk (not just this run's
        # `written`) so a partial export never clobbers the full aggregate.
        all_recs = load_all(args.results_dir)
        csv_path = write_macro_csv(all_recs, f"{args.results_dir}/macro_summary.csv")
        print(f"\nWrote {len(written)} JSON result(s) this run -> {args.results_dir}/")
        print(f"Macro summary ({len(all_recs)} total rows) -> {csv_path}")
        # Compact headline table to stdout.
        print(f"\n{'encoder':14s} {'split':6s} {'color':5s} {'approach':10s} "
              f"{'lora':4s} {'mAP':>7s}")
        for r in written:
            print(f"{r['encoder']:14s} {str(r['split']):6s} {r['color_mode']:5s} "
                  f"{r['approach']:10s} {str(r['lora']):4s} {r['macro']['mAP']:7.4f}")
    else:
        print("No results written (no matching caches found).")


if __name__ == "__main__":
    main()
