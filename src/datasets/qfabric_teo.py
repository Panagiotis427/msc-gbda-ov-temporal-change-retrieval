"""
TEOChatlas-QFabric loader — a label-grounded second dataset.

Unlike ``QFabricDataset`` (EVER-Z parquet, images-only), this reads the QFabric
crops shipped in ``jirvin16/TEOChatlas`` eval images, which come with REAL
change-type labels (the TEOChatlas RQA2 questions). Each crop is polygon-centred
(~256 px) and spans 2-5 timepoints (giving consecutive before->after pairs); the
crop's change type (one of residential / commercial / industrial / road /
demolition / mega_projects) labels EVERY pair of that crop — the crop is treated
as a single change event. This makes change-type retrieval directly
benchmarkable (Recall@K / mAP), proving the dataset-agnostic design on a second
dataset with a different taxonomy and temporal axis.

``QF_CHANGE_TYPES`` lists the canonical 6 classes (also the order used by the
query set in ``src/queries/qfabric.py``).

Filenames follow ``<loc>.d<N>.<MMDDYYYY>_<xoff>_<yoff>.tif``; a crop is keyed by
``<loc>_<xoff>_<yoff>`` and its timepoints are the distinct ``d<N>``. Labels are
read from a sidecar JSON (``crop_key -> change_type``) built by
``scripts/build_qfabric_labels.py`` from the TEOChatlas RQA2 file.
"""
from __future__ import annotations

import glob
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from PIL import Image

from .base import PairKey, PairLabel

QF_CHANGE_TYPES = ("residential", "commercial", "industrial",
                   "road", "demolition", "mega_projects")


def parse_crop(name: str) -> Optional[Tuple[str, str, str, str]]:
    """``300.d2.12262015_256_4096.tif`` -> (crop_key, loc, day, date).

    crop_key = ``<loc>_<xoff>_<yoff>`` (timepoint-independent). Returns None if
    the name doesn't match the expected scheme.
    """
    base = os.path.basename(name)
    parts = base.split(".")
    if len(parts) < 3:
        return None
    loc, day, rest = parts[0], parts[1], parts[2]
    bits = rest.split("_")
    if len(bits) < 3:
        return None
    date, xoff, yoff = bits[0], bits[1], bits[2]
    return f"{loc}_{xoff}_{yoff}", loc, day, date


def _day_index(day: str) -> int:
    return int(day[1:]) if day.startswith("d") and day[1:].isdigit() else 0


class TEOChatlasQFabricDataset:
    name = "qfabric_teo"
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
        # crop_key -> {day: filepath}
        crops: Dict[str, Dict[str, str]] = defaultdict(dict)
        for fp in glob.glob(os.path.join(self.root, "**", "*.tif"), recursive=True):
            parsed = parse_crop(fp)
            if parsed is None:
                continue
            ck, _loc, day, _date = parsed
            crops[ck][day] = fp
        # keep crops with >= 2 timepoints (a before/after pair)
        self._crops = {ck: d for ck, d in crops.items() if len(d) >= 2}

        # labels sidecar (crop_key -> change_type)
        if labels_path is None:
            cand = Path(self.root).parent / "qfabric_teo_labels.json"
            labels_path = str(cand) if cand.exists() else None
        self._labels: Dict[str, str] = (
            json.loads(Path(labels_path).read_text(encoding="utf-8")) if labels_path else {}
        )
        # keep only labelled crops
        self._crops = {ck: d for ck, d in self._crops.items() if ck in self._labels}

        # optional deterministic, stratified train/test split (for PEFT
        # generalisation: train the adapter on `train`, evaluate on held-out
        # `test`). split=None keeps the whole corpus (the §7.8 setup).
        if split in ("train", "test"):
            self._crops = self._train_test_split(split, train_frac, seed)

        # optional stratified subsample for a DEN-scale benchmark
        if max_per_class is not None:
            self._crops = self._subsample(max_per_class, seed)

        self._locations = sorted(self._crops)
        self._pairs_cache: Optional[List[PairKey]] = None

    def _train_test_split(self, split: str, train_frac: float,
                          seed: int) -> Dict[str, Dict[str, str]]:
        """Partition crops per class (stratified, seeded) into train/test —
        disjoint crop sets so no crop leaks across the split."""
        by_type: Dict[str, List[str]] = defaultdict(list)
        for ck in self._crops:
            by_type[self._labels[ck]].append(ck)
        rng = random.Random(seed)
        keep = set()
        for cks in by_type.values():
            cks = sorted(cks)
            rng.shuffle(cks)
            cut = int(round(len(cks) * train_frac))
            chosen = cks[:cut] if split == "train" else cks[cut:]
            keep.update(chosen)
        return {ck: d for ck, d in self._crops.items() if ck in keep}

    def _subsample(self, max_per_class: int, seed: int) -> Dict[str, Dict[str, str]]:
        by_type: Dict[str, List[str]] = defaultdict(list)
        for ck in self._crops:
            by_type[self._labels[ck]].append(ck)
        rng = random.Random(seed)
        keep = set()
        for t, cks in by_type.items():
            rng.shuffle(cks)
            keep.update(cks[:max_per_class])
        return {ck: d for ck, d in self._crops.items() if ck in keep}

    # -- discovery ------------------------------------------------------
    def list_locations(self) -> List[str]:
        return list(self._locations)

    def _days(self, ck: str) -> List[str]:
        return sorted(self._crops[ck], key=_day_index)

    def list_pairs(self) -> List[PairKey]:
        if self._pairs_cache is not None:
            return self._pairs_cache
        pairs: List[PairKey] = []
        for ck in self._locations:
            days = self._days(ck)
            for i in range(len(days) - 1):
                pairs.append(PairKey(location_id=ck, t1_key=days[i], t2_key=days[i + 1]))
        self._pairs_cache = pairs
        return pairs

    # -- data access ----------------------------------------------------
    def load_image(self, location_id: str, t_key: str) -> Image.Image:
        fp = self._crops[location_id][t_key]
        return Image.open(fp).convert("RGB")

    def load_pair_images(self, pair: PairKey) -> Tuple[Image.Image, Image.Image]:
        return (self.load_image(pair.location_id, pair.t1_key),
                self.load_image(pair.location_id, pair.t2_key))

    def load_metadata(self) -> pd.DataFrame:
        rows = []
        for ck in self._locations:
            for day in self._days(ck):
                rows.append({"location": ck, "timestamp": pd.Timestamp("2015-01-01")
                             + pd.Timedelta(days=_day_index(day)),
                             "t_key": day, "pair_id": f"{ck}::{day}",
                             "dataset_name": self.name})
        return pd.DataFrame(rows)

    # -- labels ---------------------------------------------------------
    def get_pair_label(self, pair: PairKey) -> Optional[PairLabel]:
        ct = self._labels.get(pair.location_id)
        if ct is None:
            return None
        # A QFabric crop is change-centred: the pair shows that change type.
        return PairLabel(change_type=ct, stable=False,
                         dominant_t1_class=ct, dominant_t2_class=ct)

    # Weak captions for PEFT supervision — phrased like the queries in
    # src/queries/qfabric.py so the adapter is pulled toward the query space.
    _CAPTIONS = {
        "residential": "new residential housing construction",
        "commercial": "new commercial buildings or retail development",
        "industrial": "new industrial facility or factory construction",
        "road": "new road or highway construction",
        "demolition": "buildings demolished, structures torn down",
        "mega_projects": "large-scale mega project under construction",
    }

    def text_caption_for_pair(self, pair: PairKey) -> str:
        ct = self._labels.get(pair.location_id)
        return self._CAPTIONS.get(ct, ct or "unknown change")
