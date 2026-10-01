"""
Publication figures from ``results/*.json`` (written by ``scripts.export_results``).

Pure consumer of the JSON results — imports only matplotlib + ``src.results_io``
(no torch / no model). Headless ``Agg`` backend; 150-dpi PNG (optionally SVG)
into ``report/figures/``.

Figures
-------
- recall curves (macro Recall@K vs K, one line per encoder x approach)
- mAP grouped bars (x=encoder, hue=approach)            <- quantitative half of
                                                            the graded comparison
- colour-mode ablation heatmap (encoder x {rgb,nrg,ndvi})
- seasonal-drift@K curves (permanent queries, lower=better); ``fig_seasonal_drift``
  is callable but not run by the CLI
- cross-split mAP (x=split, hue=approach)               <- the PEFT-overfit story

Run::

    python -m scripts.make_figures --results-dir results --out-dir report/figures
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # headless: no display, safe in CI / Windows service
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from scripts._figutil import save_fig as _save  # noqa: E402
from src.results_io import load_all  # noqa: E402

_SPLIT_ORDER = {"train": 0, "val": 1, "test": 2}


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _filter(records, *, split=None, color=None, approach=None,
            encoder=None, lora=None) -> List[Dict]:
    out = []
    for r in records:
        if split is not None and r.get("split") != split:
            continue
        if color is not None and r.get("color_mode") != color:
            continue
        if approach is not None and r.get("approach") != approach:
            continue
        if encoder is not None and r.get("encoder") != encoder:
            continue
        if lora is not None and bool(r.get("lora")) != lora:
            continue
        out.append(r)
    return out


def _approach_label(rec: Dict) -> str:
    return rec["approach"] + ("+lora" if rec.get("lora") else "")


def _ks(rec: Dict) -> List[int]:
    return [int(k) for k in rec.get("k_values", [])]


def _skip(name: str) -> None:
    print(f"  skip {name}: no matching records")


# ----------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------
def fig_recall_curves(records, out_dir, *, split="train", color="rgb",
                      svg=False) -> Optional[Path]:
    recs = _filter(records, split=split, color=color)
    if not recs:
        _skip(f"recall_curves__{split}__{color}")
        return None
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for r in sorted(recs, key=lambda x: (x["encoder"], _approach_label(x))):
        ks = _ks(r)
        ys = [r["macro"]["recall_at_k"][str(k)] for k in ks]
        ax.plot(ks, ys, marker="o", label=f"{r['encoder']} / {_approach_label(r)}")
    ax.set_xlabel("K"); ax.set_ylabel("macro Recall@K")
    ax.set_title(f"Recall@K — split={split}, color={color}")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=7)
    return _save(fig, out_dir, f"recall_curves__{split}__{color}", svg)


def _grouped_bars(ax, recs, value_fn, *, ylabel, title):
    encoders = sorted({r["encoder"] for r in recs})
    approaches = sorted({_approach_label(r) for r in recs})
    lookup = {(r["encoder"], _approach_label(r)): value_fn(r) for r in recs}
    x = np.arange(len(encoders))
    w = 0.8 / max(len(approaches), 1)
    for i, appr in enumerate(approaches):
        vals = [lookup.get((e, appr), 0.0) for e in encoders]
        ax.bar(x + i * w - 0.4 + w / 2, vals, w, label=appr)
    ax.set_xticks(x); ax.set_xticklabels(encoders, rotation=15, ha="right")
    ax.set_ylabel(ylabel); ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3); ax.legend(fontsize=8)


def fig_map_grouped_bars(records, out_dir, *, split="train", color="rgb",
                         svg=False) -> Optional[Path]:
    recs = _filter(records, split=split, color=color)
    if not recs:
        _skip(f"map_bars__{split}__{color}")
        return None
    fig, ax = plt.subplots(figsize=(7, 4.5))
    _grouped_bars(ax, recs, lambda r: r["macro"]["mAP"],
                  ylabel="mAP", title=f"mAP by encoder x approach — split={split}, color={color}")
    return _save(fig, out_dir, f"map_bars__{split}__{color}", svg)


def fig_color_ablation_heatmap(records, out_dir, *, approach="zero_shot",
                               split="test", svg=False) -> Optional[Path]:
    recs = _filter(records, split=split, approach=approach, lora=False)
    if not recs:
        _skip(f"color_ablation__{approach}__{split}")
        return None
    encoders = sorted({r["encoder"] for r in recs})
    colors = [c for c in ("rgb", "nrg", "ndvi")
              if any(r["color_mode"] == c for r in recs)]
    M = np.full((len(encoders), len(colors)), np.nan)
    for r in recs:
        if r["color_mode"] in colors:
            M[encoders.index(r["encoder"]), colors.index(r["color_mode"])] = r["macro"]["mAP"]
    fig, ax = plt.subplots(figsize=(1.6 * len(colors) + 2, 0.8 * len(encoders) + 2))
    im = ax.imshow(M, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(colors))); ax.set_xticklabels(colors)
    ax.set_yticks(range(len(encoders))); ax.set_yticklabels(encoders)
    for i in range(len(encoders)):
        for j in range(len(colors)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center",
                        color="white" if M[i, j] < np.nanmax(M) * 0.6 else "black",
                        fontsize=9)
    ax.set_title(f"mAP — colour ablation ({approach}, split={split})")
    fig.colorbar(im, ax=ax, label="mAP")
    return _save(fig, out_dir, f"color_ablation__{approach}__{split}", svg)


def fig_seasonal_drift(records, out_dir, *, split="train", color="rgb",
                       svg=False) -> Optional[Path]:
    recs = _filter(records, split=split, color=color, approach="zero_shot", lora=False)
    if not recs:
        _skip(f"seasonal_drift__{split}__{color}")
        return None
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for r in sorted(recs, key=lambda x: x["encoder"]):
        ks = _ks(r)
        ys = [r["macro"]["seasonal_drift_at_k"][str(k)] for k in ks]
        ax.plot(ks, ys, marker="s", label=r["encoder"])
    ax.set_xlabel("K"); ax.set_ylabel("seasonal drift@K (lower=better)")
    ax.set_title(f"Seasonal drift on permanent queries — split={split}, color={color}")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    return _save(fig, out_dir, f"seasonal_drift__{split}__{color}", svg)


def fig_cross_split_map(records, out_dir, *, encoder="clip_vitl14", color="rgb",
                        svg=False) -> Optional[Path]:
    recs = _filter(records, encoder=encoder, color=color)
    if not recs:
        _skip(f"cross_split__{encoder}__{color}")
        return None
    splits = sorted({r["split"] for r in recs}, key=lambda s: _SPLIT_ORDER.get(s, 9))
    approaches = sorted({_approach_label(r) for r in recs})
    lookup = {(r["split"], _approach_label(r)): r["macro"]["mAP"] for r in recs}
    x = np.arange(len(splits))
    w = 0.8 / max(len(approaches), 1)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, appr in enumerate(approaches):
        vals = [lookup.get((s, appr), 0.0) for s in splits]
        ax.bar(x + i * w - 0.4 + w / 2, vals, w, label=appr)
    ax.set_xticks(x); ax.set_xticklabels(splits)
    ax.set_ylabel("mAP")
    ax.set_title(f"Cross-split mAP — {encoder}, color={color}")
    ax.grid(True, axis="y", alpha=0.3); ax.legend(fontsize=8)
    return _save(fig, out_dir, f"cross_split__{encoder}__{color}", svg)


def fig_confusion(report: Dict, out_dir, *, svg=False) -> Optional[Path]:
    """Heatmap of a ConfusionReport dict: rows=queries, cols=actual transitions.

    The seasonal-vs-permanent error-analysis deliverable — seasonal columns
    lighting up on permanent-query rows are confusions.
    """
    labels = report.get("labels", [])
    texts = report.get("query_texts", [])
    M = np.array(report.get("matrix", []), dtype=float)
    if M.size == 0 or not labels or not texts:
        _skip("confusion (empty report)")
        return None
    fig, ax = plt.subplots(figsize=(0.7 * len(labels) + 4, 0.5 * len(texts) + 2))
    im = ax.imshow(M, cmap="magma", aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=7)
    ax.set_yticks(range(len(texts)))
    ax.set_yticklabels([t[:42] for t in texts], fontsize=7)
    vmax = M.max() if M.max() > 0 else 1.0
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if M[i, j] > 0:
                ax.text(j, i, int(M[i, j]), ha="center", va="center", fontsize=7,
                        color="white" if M[i, j] < vmax * 0.6 else "black")
    enc = report.get("encoder", "?"); appr = report.get("approach", "?")
    sp = report.get("split", "?")
    ax.set_title(f"Top-{report.get('conf_k','?')} retrieved transitions — "
                 f"{enc} / {appr} / split={sp}", fontsize=9)
    fig.colorbar(im, ax=ax, label="count")
    name = f"confusion__{enc}__{sp}__{appr}"
    return _save(fig, out_dir, name, svg)


# ----------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Generate figures from results/*.json")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out-dir", default="report/figures")
    ap.add_argument("--svg", action="store_true")
    ap.add_argument("--dataset", default="dynamic_earthnet",
                    help="the figures compare encoders and approaches within ONE dataset; the results folder "
                         "holds several (records of other datasets would overwrite the same keys)")
    ap.add_argument("--only", nargs="*", default=None,
                    help="Subset of: recall map color cross confusion (default: all)")
    args = ap.parse_args()

    records = [r for r in load_all(args.results_dir) if r.get("dataset") == args.dataset]
    if not records:
        print(f"No results in {args.results_dir}/ — run scripts.export_results first.")
        return
    out_dir = Path(args.out_dir)
    only = set(args.only) if args.only else None
    made: List[Path] = []

    def want(tag): return only is None or tag in only

    if want("recall"):
        for sp in ("train", "test"):
            p = fig_recall_curves(records, out_dir, split=sp, color="rgb", svg=args.svg)
            if p: made.append(p)
    if want("map"):
        for sp in ("train", "test"):
            p = fig_map_grouped_bars(records, out_dir, split=sp, color="rgb", svg=args.svg)
            if p: made.append(p)
    if want("color"):
        for sp in ("train", "test"):
            p = fig_color_ablation_heatmap(records, out_dir, approach="zero_shot",
                                           split=sp, svg=args.svg)
            if p: made.append(p)
    if want("cross"):
        for enc in ("clip_vitl14", "georsclip", "remoteclip"):
            p = fig_cross_split_map(records, out_dir, encoder=enc, color="rgb", svg=args.svg)
            if p: made.append(p)
        # LoRA story: georsclip NRG (zero_shot vs zero_shot+lora across splits)
        p = fig_cross_split_map(records, out_dir, encoder="georsclip", color="nrg", svg=args.svg)
        if p: made.append(p)

    if want("confusion"):
        import json
        conf_dir = Path(args.results_dir) / "confusion"
        for cf in sorted(conf_dir.glob("*.json")) if conf_dir.exists() else []:
            with open(cf, encoding="utf-8") as f:
                p = fig_confusion(json.load(f), out_dir, svg=args.svg)
            if p: made.append(p)

    print(f"\nWrote {len(made)} figure(s) -> {out_dir}/")
    for p in made:
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
