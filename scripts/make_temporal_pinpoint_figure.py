"""
Figure for the native-3 m temporal-pinpointing study
(produced by ``feature_3m_native/temporal_pinpoint.py``).

Panel A — per-query temporal mAP (ranking time-steps within each AOI timeline)
against the permutation random floor: how reliably the zero-shot change score
peaks at the true transition month rather than a stable month.
Panel B — an illustrative single-AOI timeline: the per-month change score with
the labelled transition month highlighted, showing the score spiking at the
change.

Reads the committed JSON; reproduce::

    python -m scripts.make_temporal_pinpoint_figure
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot the temporal-pinpointing study")
    ap.add_argument("--json", default="feature_3m_native/results/"
                    "temporal_pinpoint__clip_vitl14__rgb.json")
    ap.add_argument("--out-dir", default="report/figures")
    args = ap.parse_args()

    with open(args.json, encoding="utf-8") as f:
        data = json.load(f)
    rows = sorted(data["per_query"], key=lambda r: r["temporal_mAP"], reverse=True)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5))

    # Panel A — per-query temporal mAP vs random floor
    labels = [r["query"][:26] + ("…" if len(r["query"]) > 26 else "") for r in rows]
    tmap = [r["temporal_mAP"] for r in rows]
    rand = [r["rand_mAP"] for r in rows]
    sig = [r["bh_fdr"] < 0.05 for r in rows]
    y = np.arange(len(rows))
    colors = ["#27ae60" if s else "#7f8c8d" for s in sig]
    axA.barh(y, tmap, color=colors, edgecolor="black", linewidth=0.5, zorder=3)
    axA.scatter(rand, y, marker="|", s=180, color="#c0392b", zorder=4,
                label="random floor (perm.)")
    axA.set_yticks(y)
    axA.set_yticklabels(labels, fontsize=8)
    axA.invert_yaxis()
    axA.set_xlabel("temporal mAP (within-AOI step ranking)")
    axA.set_title(f"When does the change happen? — {data['encoder']}, "
                  f"{data['color_mode'].upper()}\nmacro {data['macro_temporal_mAP']} "
                  f"vs random {data['macro_rand_mAP']}; "
                  f"{data['n_fdr_significant']}/{data['n_queries_evaluable']} FDR-sig "
                  f"(green)", fontsize=9.5)
    axA.legend(loc="lower right", fontsize=8)
    axA.grid(axis="x", alpha=0.3)

    # Panel B — example AOI timeline
    ex = data.get("example_timeline")
    if ex:
        months = ex["months"]
        sc = ex["scores"]
        rstep = ex["relevant_step"]
        x = np.arange(len(months))
        axB.plot(x, sc, marker="o", color="#2980b9", lw=1.6, zorder=2)
        axB.axvline(rstep, color="#27ae60", ls="--", lw=1.8, zorder=1,
                    label=f"labelled transition ({months[rstep]})")
        axB.scatter([int(np.argmax(sc))], [max(sc)], marker="*", s=240,
                    color="#e67e22", zorder=3, label="model peak")
        step = max(1, len(months) // 8)
        axB.set_xticks(x[::step])
        axB.set_xticklabels([months[i] for i in x[::step]], rotation=45, fontsize=7,
                            ha="right")
        axB.set_ylabel("zero-shot Δ-similarity (change score)")
        axB.set_title(f"Example timeline — AOI {ex['aoi']}\n"
                      f"query: “{ex['query'][:40]}”", fontsize=9.5)
        axB.legend(loc="best", fontsize=8)
        axB.grid(alpha=0.3)
    else:
        axB.axis("off")
        axB.text(0.5, 0.5, "no single-transition example", ha="center")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"temporal_pinpoint__{data['encoder']}__{data['color_mode']}"
    fig.savefig(out_dir / f"{name}.png", dpi=150, bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.svg", bbox_inches="tight")
    print(f"wrote {out_dir / name}.png (+ .svg)")


if __name__ == "__main__":
    main()
