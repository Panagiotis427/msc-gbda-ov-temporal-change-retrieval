"""
Persist + reload benchmark results as JSON/CSV for figures and analysis.

Pure IO with **no torch import** (deliberately does not import ``src.benchmark``
at module load), so the figure / analysis scripts can consume results without
pulling a model backbone. A report is written by duck-typing on its ``to_dict``.

On-disk layout (one ``BenchmarkReport.to_dict()`` per file)::

    results/<dataset>__<encoder>__<split>__<color>__<approach>[__lora].json

plus an optional flat macro CSV aggregating many runs for quick inspection.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List


def result_path(
    results_dir: str | Path,
    dataset: str,
    encoder: str,
    split: str,
    color: str = "rgb",
    approach: str = "zero_shot",
    lora: bool = False,
    mode: str = "difference",
) -> Path:
    """Stable JSON path for one run. ``mode`` (the change-feature mode) only
    appears in the filename when it isn't the default ``difference`` — so a
    ``concatenate`` run writes distinct files and never clobbers the difference
    results."""
    mode_tag = "" if mode == "difference" else f"__{mode}"
    lora_tag = "__lora" if lora else ""
    name = f"{dataset}__{encoder}__{split}__{color}__{approach}{mode_tag}{lora_tag}.json"
    return Path(results_dir) / name


def write_report(report, path: str | Path, *, color_mode: str, split: str,
                 lora: bool = False) -> Path:
    """Serialize ``report`` (anything with ``to_dict``) to JSON.

    Idempotent: ``sort_keys=True`` + fixed indent means re-writing the same
    report produces byte-identical output.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = report.to_dict(color_mode=color_mode, split=split, lora=lora)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    return path


def read_report(path: str | Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


#: filename substrings whose JSON uses a non-standard schema (no top-level ``macro``)
#: and would otherwise flatten into a blank macro-CSV row. ``eval_rerank.py`` writes
#: nested ``strategies`` blocks; ``cv_eval.py`` writes ``full_corpus`` / ``kfold_*``.
_NON_BENCHMARK_RESULT_MARKERS = ("rerank", "cv_eval", "patch_eval", "seasonal_gate")


def load_all(results_dir: str | Path) -> List[Dict]:
    """Read every standard benchmark ``*.json`` result under *results_dir*.

    Skips sidecar files whose name contains a marker in
    ``_NON_BENCHMARK_RESULT_MARKERS`` — they use different schemas and are
    summarised by their own tooling, not the macro CSV. Empty list if absent.
    """
    results_dir = Path(results_dir)
    if not results_dir.exists():
        return []
    out: List[Dict] = []
    for p in sorted(results_dir.glob("*.json")):
        if any(m in p.name for m in _NON_BENCHMARK_RESULT_MARKERS):
            continue
        rec = read_report(p)
        # Defensive: only standard per-split benchmark reports carry "macro".
        # Any sidecar with a different schema (new marker not yet listed) is
        # skipped rather than crashing downstream figure/CSV tooling.
        if isinstance(rec, dict) and "macro" in rec:
            out.append(rec)
    return out


_CSV_FIELDS = [
    "dataset", "encoder", "split", "color_mode", "approach", "lora",
    "n_pairs", "mAP", "R@1", "R@3", "R@5", "R@10", "seasonal_drift@5",
]


def _macro_row(rec: Dict) -> Dict:
    macro = rec.get("macro", {})
    rk = macro.get("recall_at_k", {})
    sd = macro.get("seasonal_drift_at_k", {})
    return {
        "dataset": rec.get("dataset"),
        "encoder": rec.get("encoder"),
        "split": rec.get("split"),
        "color_mode": rec.get("color_mode"),
        "approach": rec.get("approach"),
        "lora": rec.get("lora"),
        "n_pairs": rec.get("n_pairs"),
        "mAP": macro.get("mAP"),
        "R@1": rk.get("1"),
        "R@3": rk.get("3"),
        "R@5": rk.get("5"),
        "R@10": rk.get("10"),
        "seasonal_drift@5": sd.get("5"),
    }


def write_macro_csv(records: List[Dict], csv_path: str | Path) -> Path:
    """Write a flat macro-metrics CSV (one row per result record), replacing any
    existing file at *csv_path*."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        w.writeheader()
        for rec in records:
            w.writerow(_macro_row(rec))
    return csv_path
