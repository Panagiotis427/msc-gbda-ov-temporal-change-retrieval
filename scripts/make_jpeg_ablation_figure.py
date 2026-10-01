"""
Figure for the controlled JPEG-vs-native 3 m ablation
(produced by ``feature_3m_native/jpeg_ablation.py``).

Plots zero-shot AOI-CV macro mAP (mean ± std over folds) as a function of JPEG
compression quality, against the native lossless raster as a horizontal upper
bound. Unlike the report's earlier native-vs-JPEG-subset table, every point here
shares the same AOIs, pairs, colour composite, encoder and folds — only the JPEG
quality changes — so the curve isolates the cost of lossy compression alone.

Reads the committed JSON; reproduce::

    python -m scripts.make_jpeg_ablation_figure
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot the JPEG-vs-native ablation curve")
    ap.add_argument("--json", default="feature_3m_native/results/"
                    "jpeg_ablation__clip_vitl14__rgb__zero_shot.json")
    ap.add_argument("--out-dir", default="report/figures")
    args = ap.parse_args()

    with open(args.json, encoding="utf-8") as f:
        data = json.load(f)
    rows = data["rows"]
    native = next(r for r in rows if r["source"] == "native")
    jpeg = sorted((r for r in rows if r["quality"] is not None),
                  key=lambda r: r["quality"])

    q = [r["quality"] for r in jpeg]
    m = [r["macro_mAP_mean"] for r in jpeg]
    s = [r["macro_mAP_std"] for r in jpeg]

    fig, ax = plt.subplots(figsize=(7, 5))

    # native lossless upper bound + its fold-variance band
    nm, ns = native["macro_mAP_mean"], native["macro_mAP_std"]
    ax.axhspan(nm - ns, nm + ns, color="#27ae60", alpha=0.12, zorder=0)
    ax.axhline(nm, ls="--", color="#27ae60", lw=1.6,
               label=f"native 3 m (lossless): {nm:.3f} ± {ns:.3f}")

    ax.errorbar(q, m, yerr=s, marker="o", capsize=4, color="#2980b9", lw=1.6,
                label="JPEG round-trip")
    for qi, mi in zip(q, m):
        ax.annotate(f"{mi:.3f}", (qi, mi), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8)

    ax.set_xlabel("JPEG quality (lower = more lossy)")
    ax.set_ylabel("DEN zero-shot AOI-CV macro mAP")
    ax.set_title(f"Native 3 m vs JPEG compression — controlled\n"
                 f"({data['encoder']}, {data['color_mode'].upper()}, "
                 f"{data['n_pairs']} pairs / {data['n_aois']} AOIs, "
                 f"{data['folds']}-fold, identical corpus)", fontsize=10)
    ax.invert_xaxis()  # quality degrades left-to-right
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8.5)

    lo = min([nm - ns] + [mi - si for mi, si in zip(m, s)])
    hi = max([nm + ns] + [mi + si for mi, si in zip(m, s)])
    ax.set_ylim(lo - 0.02, hi + 0.04)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"jpeg_ablation__{data['encoder']}__{data['color_mode']}"
    fig.savefig(out_dir / f"{name}.png", dpi=150, bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.svg", bbox_inches="tight")
    print(f"wrote {out_dir / name}.png (+ .svg)")


if __name__ == "__main__":
    main()
