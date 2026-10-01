"""Seasonal-robustness gate — stable-pair Δ-similarity false-positive rate on DEN.

Runs the image-level ``zero_shot`` change gate (``src.seasonal_gate``) over the
DEN stable pairs and reports the false-positive rate as a function of the decision
threshold. This is the direct seasonal-robustness probe that complements the
benchmark's ``seasonal_drift@K`` (N/A on the current corpora). See the report's §7 (the metric) and §8.4 (this gate).

For each change-description query ``t`` and stable pair ``(T1, T2)`` the gate scores
``Δ = cos(t, f_T2) − cos(t, f_T1)`` on whole-image embeddings; a stable pair with
``Δ > threshold`` is a false positive. Queries default to the DEN fraction-based
query set; the per-pair score is the max Δ over queries (strictest reading).

Local, deterministic (frozen encoder) — runs on the same RTX 4060 as the rest of
REPORT §8. Idempotent: a completed run is reused unless ``--force``.

Run::

    python -m scripts.run_seasonal_gate --root data/DynamicEarthNet \
        --encoder georsclip --color-mode rgb --split test
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.benchmark import encode_query
from src.datasets.dynamic_earthnet_pp import DENNpyDataset
from src.encoders import get_encoder
from src.queries.den import frac_queries
from src.seasonal_gate import evaluate_seasonal_fpr, stable_pairs


def _parse_thresholds(spec: str) -> list:
    return [float(x) for x in spec.split(",") if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Stable-pair Δ-similarity false-positive-rate seasonal gate (DEN)")
    ap.add_argument("--root", default="data/DynamicEarthNet")
    ap.add_argument("--encoder", default="georsclip")
    ap.add_argument("--color-mode", default="rgb", choices=["rgb", "nrg", "ndvi"])
    ap.add_argument("--split", default="test",
                    help="DEN split; use 'all' for every AOI")
    ap.add_argument("--thresholds", default="0.0,0.02,0.05,0.10",
                    help="comma-separated Δ decision thresholds")
    ap.add_argument("--frac-thresh", type=float, default=0.05,
                    help="fraction threshold selecting the DEN query set")
    ap.add_argument("--query", default=None,
                    help="override: a single custom change-description query")
    ap.add_argument("--prompt-ensemble", action="store_true",
                    help="ensemble the query text embedding over prompt templates")
    ap.add_argument("--results-dir", default="results")
    # Idempotency contract: results are skip-if-done; --force recomputes
    ap.add_argument("--force", action="store_true",
                    help="recompute and overwrite an existing result")
    ap.add_argument("--skip-if-done", action="store_true", default=True,
                    help="(default) reuse an existing result and log it")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and the output path, then exit")
    args = ap.parse_args()

    split = None if args.split == "all" else args.split
    split_tag = args.split
    q_tag = "customq" if args.query else f"frac{args.frac_thresh}"
    out_dir = Path(args.results_dir)
    op = out_dir / (f"seasonal_gate__{args.encoder}__{args.color_mode}"
                    f"__{split_tag}__{q_tag}.json")

    if args.dry_run:
        print(f"[dry-run] would evaluate seasonal-gate FPR -> {op}")
        print(f"[dry-run] encoder={args.encoder} color={args.color_mode} "
              f"split={split_tag} thresholds={args.thresholds} "
              f"queries={'custom' if args.query else 'frac_queries'}")
        return

    if op.exists() and not args.force:
        existing = json.loads(op.read_text(encoding="utf-8"))
        print(f"[reused] {op} already present (mode={existing.get('mode')}); "
              f"pass --force to recompute. "
              f"n_stable_pairs={existing.get('n_stable_pairs')}")
        return

    enc = get_encoder(args.encoder)
    ds = DENNpyDataset(root=args.root, split=split, color_mode=args.color_mode)
    pairs = stable_pairs(ds)
    if not pairs:
        raise RuntimeError(
            f"no stable pairs found for split={split_tag} — nothing to evaluate")

    if args.query:
        query_texts = [args.query]
    else:
        query_texts = [q.text for q in frac_queries(args.frac_thresh)]
    query_vecs = np.stack([
        encode_query(enc, text, ensemble=args.prompt_ensemble) for text in query_texts
    ]).astype(np.float32)

    thresholds = _parse_thresholds(args.thresholds)
    summary = evaluate_seasonal_fpr(ds, enc, query_vecs, thresholds, pairs=pairs)

    out = {
        "dataset": "dynamic_earthnet", "encoder": args.encoder,
        "color_mode": args.color_mode, "split": split_tag,
        "approach": "zero_shot", "metric": "stable_pair_fpr",
        "n_queries": len(query_texts), "queries": query_texts,
        "prompt_ensemble": args.prompt_ensemble,
        "thresholds": thresholds, "mode": "recomputed",
        **summary,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"[seasonal-gate] {args.encoder}/{args.color_mode}/{split_tag} — "
          f"{summary['n_stable_pairs']} stable pairs, "
          f"mean Δ-sim = {summary['mean_delta_similarity']:.4f}")
    for thr, fpr in summary["fpr_by_threshold"].items():
        print(f"  thr={thr}  FPR={fpr:.3f}")
    print(f"wrote {op}")


if __name__ == "__main__":
    main()
