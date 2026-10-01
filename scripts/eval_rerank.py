"""
Re-ranking quantification benchmark (report Appendix C) — artifact-backed.

``src/rerank.py`` post-processes a ranked result list (``diversity`` =
greedy location-deduplication; ``coherence`` = haversine proximity to the
top-1 AOI). This script measures what that re-ordering does to retrieval
quality: it ranks the pair corpus with a frozen encoder (default
GeoRSCLIP + NRG, zero_shot, test split — the §7.5 setting), then recomputes
Recall@K and mAP for the baseline order and for each re-ranking strategy,
writing one JSON artifact so the §7.5 table is reproducible.

Re-ranking is applied to the *full* ranking (``top_k = n_pairs``) so AP is
defined over a complete permutation; Recall@K reads the reranked top-K. Uses
the committed embedding cache — no GPU, no re-encode (the encoder is loaded
only to embed the handful of query strings).

CLI:
    python -m scripts.eval_rerank \
        --encoder georsclip --color-mode nrg --split test --approach zero_shot
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.benchmark import _average_precision  # noqa: E402
from src.datasets.registry import build_dataset  # noqa: E402
from src.embeddings import cache_tag_for, load_or_compute  # noqa: E402
from src.encoders import get_encoder  # noqa: E402
from src.queries import get_queries  # noqa: E402
from src.rerank import RERANK_STRATEGIES, Reranker  # noqa: E402
from src.retrieval import ChangeRetriever  # noqa: E402
from src.stats import rank_order  # noqa: E402


def _metrics(order: np.ndarray, rel: np.ndarray, k_values) -> dict:
    """Recall@K + AP for a ranking ``order`` (indices) over relevance ``rel``."""
    rel_ranked = rel[order]
    n_rel = int(rel.sum())
    recall = {int(k): float(rel[order[:k]].sum() / n_rel) for k in k_values}
    return {"recall_at_k": recall, "ap": _average_precision(rel_ranked)}


def _macro(per_query: list[dict], k_values) -> dict:
    return {
        "mAP": float(np.mean([q["ap"] for q in per_query])) if per_query else 0.0,
        "recall_at_k": {
            str(k): float(np.mean([q["recall_at_k"][k] for q in per_query]))
            for k in k_values
        },
    }


def evaluate(encoder: str, color_mode: str, split: str, approach: str,
             root: str, cache_dir: str, metadata_path: str,
             k_values=(1, 3, 5, 10), geo_weight: float = 0.3) -> dict:
    ds = build_dataset("dynamic_earthnet", root=root, pairing="bimonthly",
                       split=split, color_mode=color_mode)
    enc = get_encoder(encoder)
    store = load_or_compute(ds, enc, cache_dir=cache_dir,
                            cache_tag=cache_tag_for(split, color_mode))
    retriever = ChangeRetriever(store, enc)
    reranker = Reranker(metadata_path)
    pairs = store.pairs
    n = len(pairs)

    queries = get_queries("dynamic_earthnet")
    if not queries:
        raise ValueError("No DEN query set registered.")

    strategies = ["baseline", *RERANK_STRATEGIES]
    by_strategy: dict[str, list[dict]] = {s: [] for s in strategies}

    for q in queries:
        labels = [ds.get_pair_label(p) for p in pairs]
        rel = np.array([bool(lb is not None and q.predicate(lb)) for lb in labels])
        if rel.sum() == 0:
            continue  # not evaluable in this corpus
        scores = retriever.score_all(q.text, approach=approach)

        orders = {
            "baseline": rank_order(scores, rel),
            "diversity": reranker.rerank(scores, pairs, top_k=n, strategy="diversity"),
            "coherence": reranker.rerank(scores, pairs, top_k=n, strategy="coherence",
                                         geo_weight=geo_weight),
        }
        for s in strategies:
            m = _metrics(orders[s], rel, k_values)
            by_strategy[s].append(
                {"text": q.text, "n_relevant": int(rel.sum()), **m}
            )

    return {
        "dataset": "dynamic_earthnet",
        "encoder": enc.name,
        "approach": approach,
        "split": split,
        "color_mode": color_mode,
        "n_pairs": n,
        "k_values": [int(k) for k in k_values],
        "geo_weight": geo_weight,
        "strategies": {
            s: {"macro": _macro(by_strategy[s], k_values),
                "per_query": [
                    {**q, "recall_at_k": {str(k): v for k, v in q["recall_at_k"].items()}}
                    for q in by_strategy[s]
                ]}
            for s in strategies
        },
    }


def _print_table(report: dict) -> None:
    ks = report["k_values"]
    head = (f"\n=== Re-ranking: {report['dataset']} | {report['encoder']} | "
            f"{report['color_mode']} | {report['approach']} | {report['split']} "
            f"| N={report['n_pairs']} ===")
    print(head)
    cols = "  ".join(f"R@{k}" for k in ks)
    print(f"{'strategy':16s} {cols}  {'mAP':>6s}")
    for s, data in report["strategies"].items():
        m = data["macro"]
        r = "  ".join(f"{m['recall_at_k'][str(k)]:.3f}" for k in ks)
        print(f"{s:16s} {r}  {m['mAP']:6.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Re-ranking quantification (report Appendix C)")
    ap.add_argument("--encoder", default="georsclip")
    ap.add_argument("--color-mode", default="nrg")
    ap.add_argument("--split", default="test")
    ap.add_argument("--approach", default="zero_shot")
    ap.add_argument("--root", default="data/DynamicEarthNet")
    ap.add_argument("--cache-dir", default="data/cache")
    ap.add_argument("--metadata-path", default="data/DynamicEarthNet/aoi_metadata.json")
    ap.add_argument("--geo-weight", type=float, default=0.3)
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args()

    report = evaluate(
        encoder=args.encoder, color_mode=args.color_mode, split=args.split,
        approach=args.approach, root=args.root, cache_dir=args.cache_dir,
        metadata_path=args.metadata_path, geo_weight=args.geo_weight,
    )
    _print_table(report)

    color_tag = f"_{args.color_mode}" if args.color_mode != "rgb" else ""
    out = (Path(args.results_dir) /
           f"{report['dataset']}__{report['encoder']}__{args.split}"
           f"{color_tag}__{args.approach}__rerank.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
