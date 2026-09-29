"""
Statistical-significance audit of the committed retrieval results.

For every ``results/*.json`` benchmark report this computes, per (dataset, encoder,
split, colour, approach) cell:

* **mAP**            — the committed macro mean-AP.
* **rand_mAP**       — expected macro mAP under *random ranking* of the SAME corpus,
                       estimated by Monte-Carlo permutation (relevant set held fixed
                       per query, ranking shuffled). This is the honest baseline:
                       AP of a random ranking is ~prevalence, NOT zero. With a few
                       relevant items the estimator sits a little above bare
                       prevalence, so we simulate it rather than assume p=R/N.
* **lift**           — mAP / rand_mAP.
* **perm_p**         — one-sided permutation p-value P(rand_mAP >= mAP). Answers
                       "do the scores rank relevant items above chance?" (internal
                       validity). It does NOT answer "does the method generalise to
                       a new query" — with only 3-6 queries that is unanswerable.
* **bh**             — Benjamini-Hochberg FDR-adjusted perm_p across all held-out
                       (test/eval/val) cells, to control the ~70-cell multiple-
                       comparison problem.
* **ap_min/ap_max**  — per-query AP spread, so a single lucky query can't hide.
* **leakage**        — True for train-split PEFT/LoRA cells: the adapter was fit on
                       these exact pairs (run_pipeline.py / benchmark_qfabric.py),
                       so the number is memorisation, not retrieval.

Run::

    python -m scripts.significance_audit                 # prints table
    python -m scripts.significance_audit --csv results/results_audit_summary.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os

import numpy as np

from src.stats import perm_p_value, rand_ap

ITERS = 4000
SEED = 0


def _perm(rels, N, obs_map, rng, iters=ITERS):
    """Monte-Carlo null for the *macro* mAP (mean of per-query random APs).

    Unbiased one-sided p-value ``(#{draws >= obs} + 1) / (iters + 1)`` — the
    observed statistic counts as a null draw, so 0.0 is never returned.
    """
    draws = np.array([np.mean([rand_ap(R, N, rng) for R in rels])
                      for _ in range(iters)])
    return perm_p_value(int(np.sum(draws >= obs_map)), iters), float(draws.mean())


def collect(results_dir="results"):
    rng = np.random.default_rng(SEED)
    rows = []
    for f in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        if "confusion" in f or "rerank" in f:
            continue  # different schema; rerank is handled by eval_rerank.py
        with open(f, encoding="utf-8") as fh:
            d = json.load(fh)
        pq = d.get("per_query")
        if not pq:
            continue
        rels = [q["n_relevant"] for q in pq]
        aps = [q["ap"] for q in pq]
        N, obs = d["n_pairs"], d["macro"]["mAP"]
        p, rnd = _perm(rels, N, obs, rng)
        approach = d["approach"]
        leakage = bool(d["split"] == "train" and approach in ("peft",)) or \
            bool(d["split"] == "train" and d.get("lora"))
        rows.append({
            "dataset": d["dataset"], "encoder": d["encoder"], "split": d["split"],
            "color_mode": d["color_mode"], "approach": approach,
            "lora": bool(d.get("lora", False)), "n_queries": len(pq), "n_pairs": N,
            "mAP": round(obs, 4), "rand_mAP": round(rnd, 4),
            "lift": round(obs / rnd, 2) if rnd else 0.0,
            "perm_p": round(p, 4), "ap_min": round(min(aps), 3),
            "ap_max": round(max(aps), 3), "leakage": leakage,
        })
    # BH-FDR over held-out cells only (train cells are leakage / not a test claim).
    held = [r for r in rows if r["split"] in ("test", "eval", "val")]
    order = sorted(range(len(held)), key=lambda i: held[i]["perm_p"])
    m = len(held)
    # Benjamini-Hochberg adjusted q-values. The raw p*m/rank is not monotone in
    # rank, so the textbook step-up takes the running minimum from the largest
    # p downward (a q-value is never below a higher-ranked one) and caps at 1.
    running_min = 1.0
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        running_min = min(running_min, held[i]["perm_p"] * m / rank)
        held[i]["bh_fdr"] = round(min(1.0, running_min), 4)
    for r in rows:
        r.setdefault("bh_fdr", "")  # train cells: no FDR (not in the family)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()
    rows = collect(args.results_dir)
    cols = ["dataset", "encoder", "split", "color_mode", "approach", "lora",
            "n_queries", "n_pairs", "mAP", "rand_mAP", "lift", "perm_p",
            "bh_fdr", "ap_min", "ap_max", "leakage"]
    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote {len(rows)} rows -> {args.csv}")
    else:
        hdr = f"{'dataset':15}{'enc':11}{'spl':5}{'cm':5}{'appr':10}{'nq':>3}" \
              f"{'mAP':>7}{'rand':>6}{'lift':>5}{'p':>7}{'BH':>7} {'sig':>3} apRange"
        print(hdr)
        for r in rows:
            bh = r["bh_fdr"]
            sig = "*" if (bh != "" and bh < 0.05) else ("LEAK" if r["leakage"] else "")
            bhs = f"{bh:7.3f}" if bh != "" else f"{'--':>7}"
            print(f"{r['dataset']:15}{r['encoder']:11}{r['split']:5}{r['color_mode']:5}"
                  f"{r['approach']:10}{r['n_queries']:>3}{r['mAP']:7.3f}{r['rand_mAP']:6.3f}"
                  f"{r['lift']:5.1f}{r['perm_p']:7.3f}{bhs} {sig:>4} "
                  f"[{r['ap_min']:.2f},{r['ap_max']:.2f}]")


if __name__ == "__main__":
    main()
