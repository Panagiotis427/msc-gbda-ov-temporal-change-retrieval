"""
Build the QFabric per-timepoint development-status index from TEOChatlas RQA5.

Downloads (or reads) ``eval/QFabric_RQA5_RTQA5.json`` from ``jirvin16/TEOChatlas``
and emits ``data/QFabric/qfabric_status_labels.json`` mapping each crop's
timepoint to its real QFabric development status::

    { "<loc>_<xoff>_<yoff>": { "d1": "land_cleared", "d2": "construction_done", ... } }

These are the authentic per-timepoint statuses (the RTQA questions' answers). Each
``region_based_temporal_question_answering`` record asks "what is the development
status of this region in Image N?"; the video frames are in temporal order, so
**Image N -> the N-th day (dN)** of that crop, and the gpt answer is the status at
dN. A crop's before/after pair (di, dj) therefore encodes a *transition*
(status@di -> status@dj) — consumed by ``src.datasets.qfabric_status``.

Sibling of ``scripts/build_qfabric_labels.py`` (which builds the RQA2 *change-type*
index); this one builds the RQA5 *status-transition* index. Run::

    python -m scripts.build_qfabric_status_labels --out data/QFabric/qfabric_status_labels.json
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from src.datasets.qfabric_teo import _day_index, parse_crop

_REPO = "jirvin16/TEOChatlas"
_RQA5 = "eval/QFabric_RQA5_RTQA5.json"
_TEMPORAL_TASK = "region_based_temporal_question_answering"

# Canonical QFabric development statuses (the RTQA answer classes) -> slugs.
_STATUS = {
    "Greenland": "greenland",
    "Prior Construction": "prior_construction",
    "Land Cleared": "land_cleared",
    "Excavation": "excavation",
    "Materials Dumped": "materials_dumped",
    "Construction Started": "construction_started",
    "Construction Midway": "construction_midway",
    "Construction Done": "construction_done",
    "Operational": "operational",
}
# Longest class names first so "Construction Done" wins over a bare "Construction".
_STATUS_BY_LEN = sorted(_STATUS, key=len, reverse=True)

_ORDINALS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
}


def _gpt_answer(conversations) -> str | None:
    for c in conversations:
        if c.get("from") == "gpt":
            return (c.get("value") or "").strip()
    return None


def _match_status(ans: str | None) -> str | None:
    """Slug for the status named in *ans* (exact, else longest word-boundary match)."""
    if not ans:
        return None
    norm = ans.strip()
    for k, slug in _STATUS.items():
        if norm.lower() == k.lower():
            return slug
    for k in _STATUS_BY_LEN:
        if re.search(rf"\b{re.escape(k)}\b", ans, flags=re.IGNORECASE):
            return _STATUS[k]
    return None


def _human_question(conversations) -> str:
    for c in conversations:
        if c.get("from") == "human":
            return c.get("value") or ""
    return ""


def _image_index(question: str) -> int | None:
    """1-based frame index referenced by the question ("Image 2" / "the third image")."""
    m = re.search(r"\bImage\s+(\d+)\b", question, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(rf"\b({'|'.join(_ORDINALS)})\s+image\b", question, flags=re.IGNORECASE)
    if m:
        return _ORDINALS[m.group(1).lower()]
    return None


def build(rqa5_path: str, out_path: str) -> dict:
    data = json.loads(Path(rqa5_path).read_text(encoding="utf-8"))
    # (crop_key, day) -> Counter of status votes (a timepoint may be asked about
    # by several records / overlapping polygons). Resolve by MAJORITY VOTE.
    votes: dict[tuple[str, str], Counter] = defaultdict(Counter)
    skipped = 0
    for r in data:
        if r.get("task") != _TEMPORAL_TASK:
            continue
        vids = r.get("video") or []
        n = _image_index(_human_question(r.get("conversations", [])))
        status = _match_status(_gpt_answer(r.get("conversations", [])))
        if not vids or n is None or status is None or n > len(vids):
            skipped += 1
            continue
        parsed = parse_crop(vids[n - 1])
        if parsed is None:
            skipped += 1
            continue
        crop_key, _loc, day, _date = parsed
        votes[(crop_key, day)][status] += 1

    # Nest as {crop_key: {day: status}}; majority status per timepoint.
    nested: dict[str, dict[str, str]] = defaultdict(dict)
    n_conflict = 0
    for (ck, day), c in votes.items():
        if len(c) > 1:
            n_conflict += 1
        nested[ck][day] = c.most_common(1)[0][0]
    # Sort each crop's days for stable output.
    out = {ck: {d: days[d] for d in sorted(days, key=_day_index)}
           for ck, days in nested.items()}

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(out, indent=0))

    n_tp = sum(len(d) for d in out.values())
    print(f"Wrote {len(out)} crops / {n_tp} timepoint labels -> {out_path} "
          f"({n_conflict} timepoints had conflicting votes, resolved by majority; "
          f"{skipped} records skipped)")
    print("status distribution:", dict(Counter(s for d in out.values() for s in d.values())))
    # Transition distribution over consecutive (di, di+1) pairs (top 12).
    trans: Counter = Counter()
    for days in out.values():
        ks = sorted(days, key=_day_index)
        for i in range(len(ks) - 1):
            trans[f"{days[ks[i]]}->{days[ks[i + 1]]}"] += 1
    n_pairs = sum(trans.values())
    n_changed = sum(v for k, v in trans.items() if k.split("->")[0] != k.split("->")[1])
    print(f"consecutive pairs: {n_pairs} ({n_changed} are transitions, "
          f"{n_pairs - n_changed} stable)")
    print("top transitions:", dict(trans.most_common(12)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Build QFabric per-timepoint status labels (RQA5)")
    ap.add_argument("--rqa5", default=None,
                    help="Path to QFabric_RQA5_RTQA5.json (downloaded from TEOChatlas if omitted)")
    ap.add_argument("--out", default="data/QFabric/qfabric_status_labels.json")
    args = ap.parse_args()

    rqa5 = args.rqa5
    if rqa5 is None:
        from huggingface_hub import hf_hub_download
        rqa5 = hf_hub_download(_REPO, _RQA5, repo_type="dataset",
                               local_dir="data/QFabric/_teochatlas")
        print(f"Downloaded {rqa5}")
    build(rqa5, args.out)


if __name__ == "__main__":
    main()
