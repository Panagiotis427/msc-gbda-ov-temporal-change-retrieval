"""Qualitative retrieval examples — the salience law, shown honestly.

Renders, for two queries of contrasting salience, the actual **top-1** retrieved
bi-temporal pair as ``[Before | After | query-conditioned change heatmap]``:

* a *salient* query (e.g. new buildings) — the kind of change the frozen engine
  retrieves well (report §8.5, LEVIR-CC building AP ~0.8); and
* a *subtle* query (e.g. vegetation) — the kind that collapses toward its
  prevalence floor.

Nothing is cherry-picked: the script shows whatever pair the engine ranks first
and labels it relevant / not-relevant from the query's own label predicate, so a
weak result reads as weak. The change heatmap is included for transparency, not
as a strong localisation claim — it is a *weak* localiser (report §8.5,
pointing-game lift over the random-patch floor of -0.137 to +0.104 across encoders
and classes).

Deterministic given the cached embeddings, so it regenerates identically::

    python -m scripts.make_qualitative_figure --encoder georsclip \
        --dataset levir_mci --split test --color rgb
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.datasets.registry import build_dataset  # noqa: E402
from src.embeddings import cache_tag_for, load_or_compute  # noqa: E402
from src.encoders import get_encoder  # noqa: E402
from src.heatmap import generate_change_heatmap  # noqa: E402
from src.queries import get_queries  # noqa: E402
from src.retrieval import ChangeRetriever  # noqa: E402

_REL, _IRR = "#2ca02c", "#d62728"  # green / red — matches make_comparison_figure


def _pick_query(queries, needle: str):
    needle = needle.lower()
    for q in queries:
        if needle in q.text.lower():
            return q
    return None


def _panel(ax, arr, title: str, *, border: Optional[str] = None) -> None:
    ax.imshow(arr)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=8)
    if border is not None:
        for sp in ax.spines.values():
            sp.set_edgecolor(border); sp.set_linewidth(3)
    else:
        for sp in ax.spines.values():
            sp.set_visible(False)


def render(ds, retr: ChangeRetriever, enc, out_dir, *, dataset: str, encoder: str,
           roles, svg: bool = False) -> Optional[Path]:
    """roles: list of (label, query_needle) tuples, rendered top to bottom."""
    queries = get_queries(ds.name)
    built = []
    for label, needle in roles:
        q = _pick_query(queries, needle)
        if q is None:
            print(f"  note: no query matching '{needle}' for {ds.name} — skipped")
            continue
        results = retr.search(q.text, approach="zero_shot", top_k=1)
        if not results:
            continue
        res = results[0]
        t1, t2 = ds.load_pair_images(res.pair)
        a1 = np.asarray(t1.convert("RGB"))
        a2 = np.asarray(t2.convert("RGB"))
        _, hm = generate_change_heatmap(a1, a2, q.text, enc, alpha=0.5)
        hm_arr = np.asarray(hm.convert("RGB")) if hm is not None else a2
        lb = ds.get_pair_label(res.pair)
        relevant = bool(lb is not None and q.predicate(lb))
        built.append((label, q, res, a1, a2, hm_arr, relevant))

    if not built:
        print("  skip qualitative figure: no queries resolved")
        return None

    fig, axes = plt.subplots(len(built), 4, figsize=(10.0, 2.9 * len(built) + 0.8),
                             gridspec_kw={"width_ratios": [0.9, 1, 1, 1]},
                             squeeze=False)
    for i, (label, q, res, a1, a2, hm_arr, relevant) in enumerate(built):
        color = _REL if relevant else _IRR
        verdict = "top-1 relevant" if relevant else "top-1 not relevant"
        tax = axes[i][0]
        tax.axis("off")
        tax.text(0.98, 0.5,
                 f"{label}\n\"{q.text}\"\nscore {res.score:.2f}\n{verdict}",
                 ha="right", va="center", fontsize=8.5, color=color,
                 transform=tax.transAxes)
        _panel(axes[i][1], a1, "Before (T1)")
        _panel(axes[i][2], a2, "After (T2)")
        _panel(axes[i][3], hm_arr, "Query-conditioned change heatmap", border=color)

    fig.suptitle(
        f"Qualitative retrieval: salient vs subtle change ({encoder}, {dataset})\n"
        "Actual top-1 zero-shot pair per query [Before | After | change heatmap]; "
        "green = relevant, red = not (query label predicate).\n"
        "The change heatmap is a weak localiser (report Sec. 8.5), shown for transparency.",
        fontsize=9)
    fig.tight_layout(rect=[0.01, 0, 1, 0.93])
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    name = f"qualitative_examples__{encoder}"
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    if svg:
        fig.savefig(out_dir / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description="Qualitative salient-vs-subtle retrieval figure")
    ap.add_argument("--dataset", default="levir_mci")
    ap.add_argument("--root", default="data/_levir_mci/extracted/LEVIR-MCI-dataset")
    ap.add_argument("--encoder", default="georsclip",
                    choices=["clip_vitl14", "georsclip", "remoteclip"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--color", default="rgb", choices=["rgb", "nrg", "ndvi"])
    ap.add_argument("--pairing", default="bimonthly")
    ap.add_argument("--cache-dir", default="data/cache")
    ap.add_argument("--out-dir", default="report/figures")
    # Which queries play the salient (success) and subtle (failure) roles; matched
    # as substrings against the dataset's query texts.
    ap.add_argument("--salient", default="building")
    ap.add_argument("--subtle", default="vegetation")
    ap.add_argument("--svg", action="store_true")
    args = ap.parse_args()

    enc = get_encoder(args.encoder)
    ds = build_dataset(args.dataset, root=args.root, pairing=args.pairing,
                       split=args.split, color_mode=args.color)
    store = load_or_compute(ds, enc, cache_dir=args.cache_dir,
                            cache_tag=cache_tag_for(args.split, args.color))
    retr = ChangeRetriever(store, enc)
    roles = [("Salient change", args.salient), ("Subtle change", args.subtle)]
    path = render(ds, retr, enc, args.out_dir, dataset=args.dataset,
                  encoder=args.encoder, roles=roles, svg=args.svg)
    if path:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
