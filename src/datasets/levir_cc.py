"""
LEVIR-CC loader --- bi-temporal building-change pairs with human change captions.

LEVIR-CC (Liu et al., IEEE TGRS 2022) provides 10,077 bi-temporal remote-sensing
image pairs (256x256, 0.5 m/pixel), each annotated with five natural-language
change captions and a binary change flag. Layout after download (HuggingFace
``lcybuaa/LEVIR-CC``)::

    <root>/
      LevirCCcaptions.json
      images/{train,val,test}/A/<name>.png   # pre-phase  (T1)
      images/{train,val,test}/B/<name>.png   # post-phase (T2)

Unlike DEN (LULC-derived weak labels) and QFabric (categorical RQA2 types), this
is an open-vocabulary corpus driven by real human change descriptions: each pair
carries change tags parsed from its captions, and ``src/queries/levir_cc.py``
maps free-text queries to those tags. Implements the ``TemporalDataset``
protocol, so the rest of the pipeline (embeddings / retrieval / benchmark / app)
is unchanged.
"""
from __future__ import annotations

import json
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from PIL import Image

from .base import PairKey, PairLabel

# Caption keyword -> change tag. The tag set is the open-vocabulary change-type
# proxy that the query relevance rules (src/queries/levir_cc.py) test against.
_TAG_RULES: Dict[str, re.Pattern] = {
    "building": re.compile(r"\b(building|buildings|villa|villas|house|houses|"
                           r"residential|apartment|cottage|roof)\b", re.I),
    "road": re.compile(r"\b(road|roads|street|streets|path|driveway)\b", re.I),
    "demolition": re.compile(r"\b(disappear|disappears|disappeared|removed|"
                             r"demolish|demolished|razed|torn down)\b", re.I),
    "vegetation": re.compile(r"\b(forest|trees|tree|vegetation|grass|woodland|"
                             r"farmland|vegetated)\b", re.I),
    "water": re.compile(r"\b(water|lake|pond|river|reservoir)\b", re.I),
}


class LevirCCDataset:
    """``TemporalDataset`` over LEVIR-CC image pairs and their change captions."""

    name = "levir_cc"
    temporal_axis_type = "pair"

    def __init__(self, root, split: Optional[str] = None, **_ignore):
        if _ignore:
            # Extra keywords stay accepted (one option set can feed several loaders),
            # but a misspelled argument must not disappear without a trace.
            warnings.warn(f"{type(self).__name__}: ignoring unsupported argument(s) "
                          f"{sorted(_ignore)}", stacklevel=2)
        self.root = Path(root)
        cap = self.root / "LevirCCcaptions.json"
        if not cap.exists():
            raise FileNotFoundError(
                f"LevirCCcaptions.json not found under {self.root}. Download LEVIR-CC "
                "(HuggingFace lcybuaa/LEVIR-CC) into this directory first.")
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
            caps = [(s.get("raw") or s.get("sentence") or "").strip()
                    for s in img.get("sentences", [])]
            flag = int(img.get("changeflag", 1))
            loc = Path(fname).stem
            self._records[loc] = {
                "split": sp, "filename": fname, "captions": caps,
                "tags": self._tags(caps, flag), "stable": flag == 0,
            }
        self._locations = sorted(self._records)
        self._split = split
        if split and not self._records:
            warnings.warn(f"{type(self).__name__}: no pairs found for split={split!r} "
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
    def _img_path(self, loc: str, ab: str) -> Path:
        r = self._records[loc]
        return self.root / "images" / r["split"] / ab / r["filename"]

    def load_image(self, location_id: str, t_key: str) -> Image.Image:
        ab = "A" if t_key in ("A", "t1") else "B"
        return Image.open(self._img_path(location_id, ab)).convert("RGB")

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
        """First human caption --- used as PEFT weak supervision if ever trained."""
        caps = self.captions_for(pair.location_id)
        return caps[0] if caps else "remote sensing land cover change"
