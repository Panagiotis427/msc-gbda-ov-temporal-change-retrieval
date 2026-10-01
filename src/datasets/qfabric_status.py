"""
TEOChatlas-QFabric *status-transition* loader (RQA5).

Sibling of :class:`~src.datasets.qfabric_teo.TEOChatlasQFabricDataset` (RQA2
change-*type*). Same crops, but labels are the **per-timepoint development
status** (built by ``scripts/build_qfabric_status_labels.py`` from RQA5/RTQA5):
a nested sidecar ``{crop_key: {day: status}}`` over the 9 QFabric statuses.

A before/after pair therefore carries a *transition* ``status@t1 -> status@t2``,
exposed via ``PairLabel.dominant_t1_class`` / ``dominant_t2_class`` so the
existing ``benchmark._transition`` predicates and ``src/queries/qfabric_status``
can target src->dst transitions. Status is a temporal progression, so this is
the QFabric task where the directional ``zero_shot`` Δ-signal is expected to
beat ``naive`` (cos(text, f_T2)) — the opposite regime to change-type (report §8.5).

Crop discovery / pairing / image loading are inherited unchanged; only the
label model, ``get_pair_label``, PEFT captions, and the (per-crop, by
final-status) stratification are overridden.
"""
from __future__ import annotations

import glob
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from .base import PairKey, PairLabel
from .qfabric_teo import TEOChatlasQFabricDataset, parse_crop

QF_STATUSES = (
    "greenland", "prior_construction", "land_cleared", "excavation",
    "materials_dumped", "construction_started", "construction_midway",
    "construction_done", "operational",
)


class StatusQFabricDataset(TEOChatlasQFabricDataset):
    name = "qfabric_status"
    temporal_axis_type = "pair"

    def __init__(
        self,
        root: str,
        labels_path: Optional[str] = None,
        max_per_class: Optional[int] = None,
        seed: int = 42,
        split: Optional[str] = None,
        train_frac: float = 0.8,
    ) -> None:
        self.root = str(root)
        # crop_key -> {day: filepath} (same discovery as the RQA2 loader)
        crops: Dict[str, Dict[str, str]] = defaultdict(dict)
        for fp in glob.glob(os.path.join(self.root, "**", "*.tif"), recursive=True):
            parsed = parse_crop(fp)
            if parsed is None:
                continue
            ck, _loc, day, _date = parsed
            crops[ck][day] = fp
        self._crops = {ck: d for ck, d in crops.items() if len(d) >= 2}

        # nested per-timepoint status sidecar: {crop_key: {day: status}}
        if labels_path is None:
            cand = Path(self.root).parent / "qfabric_status_labels.json"
            labels_path = str(cand) if cand.exists() else None
        self._status: Dict[str, Dict[str, str]] = (
            json.loads(Path(labels_path).read_text(encoding="utf-8")) if labels_path else {}
        )
        # keep crops with >= 2 status-labelled timepoints (>= 1 labelled transition)
        self._crops = {
            ck: d for ck, d in self._crops.items()
            if len(set(d) & set(self._status.get(ck, {}))) >= 2
        }

        # crop-level (no-leakage) split + stratified subsample, keyed by the
        # crop's FINAL labelled status (a crop has many per-timepoint statuses,
        # so there is no single per-crop label as in RQA2).
        if split in ("train", "test"):
            self._crops = self._train_test_split(split, train_frac, seed)
        if max_per_class is not None:
            self._crops = self._subsample(max_per_class, seed)

        self._locations = sorted(self._crops)
        self._pairs_cache: Optional[List[PairKey]] = None

    # -- stratification -------------------------------------------------
    def _crop_class(self, ck: str) -> str:
        """Per-crop stratification key = status at the last labelled timepoint."""
        labelled = [d for d in self._days(ck) if d in self._status.get(ck, {})]
        return self._status[ck][labelled[-1]] if labelled else "unknown"

    def _by_class(self) -> Dict[str, List[str]]:
        bt: Dict[str, List[str]] = defaultdict(list)
        for ck in self._crops:
            bt[self._crop_class(ck)].append(ck)
        return bt

    def _train_test_split(self, split: str, train_frac: float,
                          seed: int) -> Dict[str, Dict[str, str]]:
        rng = random.Random(seed)
        keep = set()
        for cks in self._by_class().values():
            cks = sorted(cks)
            rng.shuffle(cks)
            cut = int(round(len(cks) * train_frac))
            keep.update(cks[:cut] if split == "train" else cks[cut:])
        return {ck: d for ck, d in self._crops.items() if ck in keep}

    def _subsample(self, max_per_class: int, seed: int) -> Dict[str, Dict[str, str]]:
        rng = random.Random(seed)
        keep = set()
        for cks in self._by_class().values():
            cks = sorted(cks)
            rng.shuffle(cks)
            keep.update(cks[:max_per_class])
        return {ck: d for ck, d in self._crops.items() if ck in keep}

    # -- labels ---------------------------------------------------------
    def get_pair_label(self, pair: PairKey) -> Optional[PairLabel]:
        st = self._status.get(pair.location_id)
        if not st:
            return None
        s1, s2 = st.get(pair.t1_key), st.get(pair.t2_key)
        if s1 is None or s2 is None:
            return None
        return PairLabel(change_type=f"{s1}->{s2}", stable=(s1 == s2),
                         dominant_t1_class=s1, dominant_t2_class=s2)

    # Weak captions for PEFT supervision, keyed by the achieved (t2) status and
    # phrased like the queries in src/queries/qfabric_status.py.
    _CAPTIONS = {
        "greenland": "undeveloped vacant green land",
        "prior_construction": "site before any construction",
        "land_cleared": "land cleared and prepared for development",
        "excavation": "excavation and earthworks on the site",
        "materials_dumped": "construction materials dumped on the site",
        "construction_started": "new building construction has started",
        "construction_midway": "a building under active construction",
        "construction_done": "construction completed, finished buildings",
        "operational": "a completed development now operational",
    }

    def text_caption_for_pair(self, pair: PairKey) -> str:
        # Weak PEFT caption keyed on the achieved (t2) status only — a
        # deliberate end-state framing: it matches the query phrasing in
        # src/queries/qfabric_status.py and keeps captions low-cardinality. A
        # consequence is that a stable pair (status unchanged) and a
        # transitioning pair that ends at the same status share a caption, so
        # the PEFT supervision carries some change-vs-stable label noise. This
        # is intentional and only affects the PEFT approach; the headline
        # directional signal here is zero_shot (cos Δ), which ignores captions.
        st = self._status.get(pair.location_id, {})
        s2 = st.get(pair.t2_key)
        if s2 is None:
            return "satellite image of a location"
        return self._CAPTIONS.get(s2, f"land changed to {s2.replace('_', ' ')}")
