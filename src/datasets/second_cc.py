"""
SECOND-CC loader --- bi-temporal land-cover-change pairs with human change
captions *and* pixel-level semantic maps for both phases.

SECOND-CC (Robust Change Captioning in Remote Sensing, arXiv:2501.10075) pairs
6,041 bi-temporal RS scenes (256x256) with 30,205 human change captions and the
six-class SECOND semantic maps for T1 and T2. It is the open-vocabulary *breadth*
counterpart to LEVIR-CC (whose change is almost entirely building/road): SECOND-CC
spans tree, low-vegetation, water, ground, building and playground change, so its
captions exercise a far wider change vocabulary.

Layout after extraction (Zenodo ``10.5281/zenodo.16937571``)::

    <root>/                                   # .../SECOND-CC-AUG
      SECOND-CC-AUG.json
      {train,val,test}/rgb/A/<id>.png         # pre-phase  (T1)
      {train,val,test}/rgb/B/<id>.png         # post-phase (T2)
      {train,val,test}/sem/A/<id>.png         # T1 semantic map (6-class, RGB-coded)
      {train,val,test}/sem/B/<id>.png         # T2 semantic map

The caption JSON mirrors the LEVIR-CC schema (``{"images": [{"filename",
"split", "changeflag", "sentences": [{"raw", "tokens", ...}]}]}``), so retrieval
relevance is derived exactly as in ``levir_cc``: each pair carries change tags
parsed from its captions, and ``src/queries/second_cc.py`` maps free-text queries
to those tags.

Beyond retrieval, the per-phase semantic maps make this loader the richest
localization source in the project: :meth:`load_change_mask` gives a per-class
change mask (T2-class basis) and :meth:`transition_change_mask` gives a genuine
``from -> to`` land-cover transition mask (the directed transitions among the six
classes are the dataset's "30 change categories"). Consumed by
``scripts/eval_localization.py``.

Wired through ``src/datasets/registry.py`` (``second_cc``); retrieval queries in
``src/queries/second_cc.py``.
"""
from __future__ import annotations

import json
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

from .base import PairKey, PairLabel

# SECOND six-class semantic palette (RGB -> class), verified against the extracted
# sem PNGs. White is no-change / no-data. Self-contained constants.
NO_CHANGE_RGB = (255, 255, 255)
CLASS_RGB: Dict[str, Tuple[int, int, int]] = {
    "ground":         (128, 128, 128),   # non-vegetated ground surface
    "tree":           (0, 255, 0),
    "low_vegetation": (0, 128, 0),
    "water":          (0, 0, 255),
    "building":       (128, 0, 0),
    "playground":     (255, 0, 0),
}
# class -> working index (1..6); 0 = no-change / no-data.
CLASS_TO_INDEX: Dict[str, int] = {c: i + 1 for i, c in enumerate(CLASS_RGB)}

# Caption keyword -> change tag. Tags are the open-vocab change-type proxy the
# query relevance rules (src/queries/second_cc.py) test against. "road" is carried
# because SECOND-CC captions mention it heavily even though it is not one of the
# six semantic-map classes (so road has retrieval relevance but no localization
# mask -- stated honestly, mirroring LEVIR-MCI's query/mask coverage gap).
_TAG_RULES: Dict[str, re.Pattern] = {
    "building":       re.compile(r"\b(building|buildings|house|houses|structure|"
                                 r"structures|residential|villa|villas)\b", re.I),
    "road":           re.compile(r"\b(road|roads|street|streets|path|pathway)\b", re.I),
    "tree":           re.compile(r"\b(tree|trees|forest|woodland)\b", re.I),
    "low_vegetation": re.compile(r"\b(vegetation|grass|grassland|meadow|crop|crops|"
                                 r"farmland|lawn)\b", re.I),
    "water":          re.compile(r"\b(water|lake|pond|river|reservoir|pool)\b", re.I),
    "ground":         re.compile(r"\b(bareland|bare\s*land|bare\s*ground|soil|"
                                 r"ground|barren)\b", re.I),
    "playground":     re.compile(r"\b(playground|sports?\s*field|court|stadium)\b", re.I),
}

# Retrieval-query text -> semantic-map class for localization (the six masked
# classes; road has no mask).
QUERY_TO_MASK_CLASS: Dict[str, str] = {
    "new buildings or structures appeared": "building",
    "trees appeared or were cleared": "tree",
    "low vegetation or grassland changed": "low_vegetation",
    "a water body appeared or changed": "water",
    "bare ground or land cleared": "ground",
    "a playground or sports field": "playground",
}


class SecondCCDataset:
    """``TemporalDataset`` over SECOND-CC image pairs, captions and semantic maps."""

    name = "second_cc"
    temporal_axis_type = "pair"

    def __init__(self, root, split: Optional[str] = None, **_ignore):
        if _ignore:
            # Extra keywords stay accepted (one option set can feed several loaders),
            # but a misspelled argument must not disappear without a trace.
            warnings.warn(f"{type(self).__name__}: ignoring unsupported argument(s) "
                          f"{sorted(_ignore)}", stacklevel=2)
        self.root = Path(root)
        cap = self.root / "SECOND-CC-AUG.json"
        if not cap.exists():
            raise FileNotFoundError(
                f"SECOND-CC-AUG.json not found under {self.root}. Download SECOND-CC "
                "(Zenodo 10.5281/zenodo.16937571) and extract into this directory first.")
        data = json.loads(cap.read_text(encoding="utf-8"))
        images = data["images"] if isinstance(data, dict) else data
        self._records: Dict[str, dict] = {}
        for img in images:
            sp = img.get("split") or img.get("filepath") or "train"
            if split and sp != split:
                continue
            fname = img.get("filename") or img.get("file_name")
            if not fname:
                continue
            caps = [(s.get("raw") or "").strip() for s in img.get("sentences", [])]
            flag = int(img.get("changeflag", 1))
            loc = Path(fname).stem
            self._records[loc] = {
                "split": sp, "filename": fname, "captions": caps,
                "tags": self._tags(caps, flag), "stable": flag == 0,
            }
        self._locations = sorted(self._records)
        self._split = split
        if split and not self._records:
            warnings.warn(f"SecondCCDataset: no pairs found for split={split!r} "
                          f"under {self.root} — check the split name / layout.")

    @staticmethod
    def _tags(captions: List[str], flag: int) -> List[str]:
        if flag == 0:
            return ["stable"]
        text = " ".join(captions)
        tags = [t for t, rx in _TAG_RULES.items() if rx.search(text)]
        return tags or ["change"]

    # -- discovery ------------------------------------------------------
    def list_locations(self) -> List[str]:
        return list(self._locations)

    def list_pairs(self) -> List[PairKey]:
        return [PairKey(location_id=loc, t1_key="A", t2_key="B")
                for loc in self._locations]

    # -- data access ----------------------------------------------------
    def _rgb_path(self, loc: str, ab: str) -> Path:
        r = self._records[loc]
        return self.root / r["split"] / "rgb" / ab / r["filename"]

    def _sem_path(self, loc: str, ab: str) -> Path:
        r = self._records[loc]
        return self.root / r["split"] / "sem" / ab / r["filename"]

    def load_image(self, location_id: str, t_key: str) -> Image.Image:
        ab = "A" if t_key in ("A", "t1") else "B"
        return Image.open(self._rgb_path(location_id, ab)).convert("RGB")

    def load_pair_images(self, pair: PairKey) -> Tuple[Image.Image, Image.Image]:
        return (self.load_image(pair.location_id, "A"),
                self.load_image(pair.location_id, "B"))

    def load_metadata(self) -> pd.DataFrame:
        rows = []
        for loc in self._locations:
            for tk, ts in (("A", "2017-01-01"), ("B", "2020-01-01")):
                rows.append({"location": loc, "timestamp": pd.Timestamp(ts),
                             "t_key": tk, "pair_id": f"{loc}::{tk}",
                             "dataset_name": self.name})
        return pd.DataFrame(rows)

    # -- labels ---------------------------------------------------------
    def get_pair_label(self, pair: PairKey) -> Optional[PairLabel]:
        r = self._records.get(pair.location_id)
        if r is None:
            return None
        return PairLabel(change_type="|".join(r["tags"]), stable=r["stable"])

    def captions_for(self, location_id: str) -> List[str]:
        r = self._records.get(location_id)
        return list(r["captions"]) if r else []

    def text_caption_for_pair(self, pair: PairKey) -> str:
        caps = self.captions_for(pair.location_id)
        return caps[0] if caps else "remote sensing land cover change"

    # -- semantic-map change masks (localization) -----------------------
    def _decode_sem(self, path: Path) -> np.ndarray:
        """RGB semantic map -> [H,W] class-index array (0 = no-change/no-data)."""
        arr = np.array(Image.open(path).convert("RGB"))
        idx = np.zeros(arr.shape[:2], dtype=np.uint8)
        for cls, rgb in CLASS_RGB.items():
            idx[np.all(arr == np.array(rgb, dtype=arr.dtype), axis=-1)] = CLASS_TO_INDEX[cls]
        return idx

    def _sem_pair(self, pair: PairKey) -> Tuple[np.ndarray, np.ndarray]:
        return (self._decode_sem(self._sem_path(pair.location_id, "A")),
                self._decode_sem(self._sem_path(pair.location_id, "B")))

    def load_change_mask(self, pair: PairKey,
                         change_class: Optional[str] = None) -> np.ndarray:
        """Change mask from the two semantic maps.

        ``change_class=None`` -> ``uint8`` class-index map of the change: changed
        pixels carry their **T2** class index (1..6 per :data:`CLASS_TO_INDEX`),
        unchanged pixels are 0. (Same contract as ``LevirMCIDataset.load_change_mask``:
        ``None`` returns a class-index map, a class name returns a boolean mask.)
        A class name -> boolean mask of changed pixels whose **T2** class is that
        class (the "appeared as" basis, matching the localization queries)."""
        l1, l2 = self._sem_pair(pair)
        changed = l1 != l2
        if change_class is None:
            return np.where(changed, l2, 0).astype(np.uint8)
        if change_class not in CLASS_TO_INDEX:
            raise ValueError(f"unknown class {change_class!r}; expected one of "
                             f"{sorted(CLASS_TO_INDEX)}")
        return changed & (l2 == CLASS_TO_INDEX[change_class])

    def transition_change_mask(self, pair: PairKey,
                               from_cls: str, to_cls: str) -> np.ndarray:
        """Directed ``from_cls -> to_cls`` transition mask (T1==from & T2==to).

        The directed transitions among the six classes are the dataset's "30
        change categories" (6x5 ordered pairs)."""
        for c in (from_cls, to_cls):
            if c not in CLASS_TO_INDEX:
                raise ValueError(f"unknown class {c!r}; expected one of "
                                 f"{sorted(CLASS_TO_INDEX)}")
        l1, l2 = self._sem_pair(pair)
        return (l1 == CLASS_TO_INDEX[from_cls]) & (l2 == CLASS_TO_INDEX[to_cls])

    def has_mask(self, location_id: str) -> bool:
        if location_id not in self._records:
            return False
        return self._sem_path(location_id, "B").exists()
