"""
Dynamic EarthNet — *preprocessed* (DynNet gdown) layout loader.

This is the format of the ~7 GB gdown subset (different from the raster
``planet/<aoi>/<date>.tif`` layout handled by ``DENDataset``):

    <root>/
    ├── labels/<AOI>.npy                 # uint8 [24, 1024, 1024] monthly LULC
    ├── split.json                       # {"train":[...], "val":[...], "test":[...]}
    ├── train/<AOI>/<AOI>_<i>_rgb.jpeg   # ~730 daily RGB frames (1024²)
    ├── train/<AOI>/<AOI>_<i>_infra.jpeg # ~730 daily NIR frames (1024², grayscale)
    ├── val/<AOI>/...
    └── test/<AOI>/...

24 monthly label maps are the change-detection timeline. Each month ``m`` is
mapped to a representative daily RGB frame by spreading evenly across the
~730 daily frames, so a bi-temporal pair (month m1, m2) has both imagery and a
ground-truth ``PairLabel`` (reusing ``derive_pair_label``).

Implements the ``TemporalDataset`` protocol; the rest of the pipeline
(embeddings / retrieval / benchmark / app) is unchanged.

``color_mode`` controls how images are loaded:
- ``'rgb'``  — standard 3-channel RGB (default, CLIP-compatible).
- ``'nrg'``  — NIR-Red-Green false colour; maps vegetation strongly (NDVI-like).
              Composed from ``_infra.jpeg`` (NIR) + R/G channels of ``_rgb.jpeg``.
- ``'ndvi'`` — single-channel NDVI replicated to 3 channels for CLIP input.
              NDVI = (NIR - R) / (NIR + R + ε), stretched to [0, 255].
"""
from __future__ import annotations

import glob
import json
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

from .base import PairKey, PairLabel
from .dynamic_earthnet import CLASS_NAMES, derive_pair_label

_N_MONTHS = 24


def resolve_pp_root(root: str | Path) -> Optional[Path]:
    """Return the dir that actually holds labels/ + split.json, descending one
    level if the archive double-nested (``DynamicEarthNet/DynamicEarthNet``)."""
    root = Path(root)
    for cand in (root, root / "DynamicEarthNet"):
        if (cand / "labels").is_dir() and list((cand / "labels").glob("*.npy")):
            return cand
    return None


def _month_selection(strategy: str) -> List[int]:
    months = list(range(_N_MONTHS))
    if strategy == "monthly":
        return months
    if strategy == "bimonthly":
        return months[::2]
    if strategy == "seasonal-quartet":
        # One frame per season within the first year (≈ Jan / Apr / Jul / Oct):
        # four distinct seasons. The previous ``months[::6]`` gave [0, 6, 12, 18],
        # i.e. only winter/summer repeated across the 2-year span — not one per
        # season, contradicting the name.
        return [0, 3, 6, 9]
    raise ValueError(f"Unknown pairing_strategy: {strategy!r}")


_VALID_COLOR_MODES = {"rgb", "nrg", "ndvi"}


def _compose_nrg(rgb_path: Path, infra_path: Path) -> Image.Image:
    """NIR-Red-Green false-colour composite (3-channel, CLIP-compatible)."""
    rgb = np.array(Image.open(rgb_path).convert("RGB"))
    if infra_path.exists():
        nir = np.array(Image.open(infra_path).convert("L"))
    else:
        warnings.warn(
            f"NIR frame missing ({infra_path.name}); NRG falls back to the red "
            "channel, yielding a (R,R,G) image — colour-mode results will be "
            "invalid for this frame.", RuntimeWarning, stacklevel=2,
        )
        nir = rgb[:, :, 0]
    nrg = np.stack([nir, rgb[:, :, 0], rgb[:, :, 1]], axis=-1)
    return Image.fromarray(nrg.astype(np.uint8))


def _compose_ndvi(rgb_path: Path, infra_path: Path) -> Image.Image:
    """NDVI single-band replicated to 3 channels (float → uint8, 0-centred)."""
    rgb = np.array(Image.open(rgb_path).convert("RGB")).astype(np.float32)
    red = rgb[:, :, 0]
    if infra_path.exists():
        nir = np.array(Image.open(infra_path).convert("L")).astype(np.float32)
    else:
        warnings.warn(
            f"NIR frame missing ({infra_path.name}); NDVI falls back to the red "
            "channel, yielding an all-zero index — colour-mode results will be "
            "invalid for this frame.", RuntimeWarning, stacklevel=2,
        )
        nir = red
    ndvi = (nir - red) / (nir + red + 1e-6)
    ndvi_u8 = np.clip((ndvi + 1.0) * 127.5, 0, 255).astype(np.uint8)
    out = np.stack([ndvi_u8, ndvi_u8, ndvi_u8], axis=-1)
    return Image.fromarray(out)


class DENNpyDataset:
    """Dynamic EarthNet preprocessed (npy) loader — ``TemporalDataset``."""

    CLASS_NAMES = CLASS_NAMES
    name = "dynamic_earthnet"
    temporal_axis_type = "daily"

    def __init__(
        self,
        root: str | Path,
        pairing_strategy: str = "bimonthly",
        stable_threshold: float = 0.02,
        split: Optional[str] = "test",
        aoi_filter: Optional[List[str]] = None,
        color_mode: str = "rgb",
    ) -> None:
        if color_mode not in _VALID_COLOR_MODES:
            raise ValueError(f"color_mode must be one of {_VALID_COLOR_MODES}, got {color_mode!r}")
        self.color_mode = color_mode
        resolved = resolve_pp_root(root)
        if resolved is None:
            raise FileNotFoundError(
                f"No preprocessed DEN (labels/*.npy) found under {root}")
        self.root = resolved
        self.pairing_strategy = pairing_strategy
        self.stable_threshold = stable_threshold
        self._labels_dir = self.root / "labels"

        split_map: Dict[str, str] = {}
        sj = self.root / "split.json"
        if sj.exists():
            for sp, aois in json.loads(sj.read_text()).items():
                for a in aois:
                    split_map[a] = sp
        self._split_map = split_map

        all_aois = sorted(p.stem for p in self._labels_dir.glob("*.npy"))
        if split and split_map:
            all_aois = [a for a in all_aois if split_map.get(a) == split]
        if aoi_filter:
            all_aois = [a for a in all_aois if a in aoi_filter]
        self._aois = all_aois

        self._months = _month_selection(pairing_strategy)
        self._label_cache: Dict[str, np.ndarray] = {}
        self._ndaily_cache: Dict[str, int] = {}
        self._pairs = self._build_pairs()

    # -- TemporalDataset protocol --------------------------------------
    def list_locations(self) -> List[str]:
        return list(self._aois)

    def list_pairs(self) -> List[PairKey]:
        return list(self._pairs)

    def _aoi_dir(self, aoi: str) -> Path:
        sp = self._split_map.get(aoi)
        if sp and (self.root / sp / aoi).is_dir():
            return self.root / sp / aoi
        for sp in ("train", "val", "test"):
            if (self.root / sp / aoi).is_dir():
                return self.root / sp / aoi
        raise FileNotFoundError(f"No imagery dir for AOI {aoi}")

    def _n_daily(self, aoi: str) -> int:
        if aoi not in self._ndaily_cache:
            d = self._aoi_dir(aoi)
            n = len(glob.glob(str(d / f"{aoi}_*_rgb.jpeg")))
            self._ndaily_cache[aoi] = max(n, 1)
        return self._ndaily_cache[aoi]

    def _month_to_didx(self, aoi: str, month: int) -> int:
        n = self._n_daily(aoi)
        return int(round(month * (n - 1) / (_N_MONTHS - 1)))

    def _resolve_frame_path(self, location_id: str, didx: int, suffix: str,
                             required: bool = True) -> Optional[Path]:
        d = self._aoi_dir(location_id)
        path = d / f"{location_id}_{didx}_{suffix}.jpeg"
        if path.exists():
            return path
        def _fnum(p: str) -> int:
            return int(re.search(rf"_(\d+)_{suffix}", p).group(1))

        cands = sorted(glob.glob(str(d / f"{location_id}_*_{suffix}.jpeg")), key=_fnum)
        if cands:
            # Pick the candidate whose actual frame NUMBER is nearest to didx.
            # (The old positional ``cands[min(didx, len-1)]`` silently returned the
            # wrong frame whenever the frame numbering had gaps.)
            return Path(min(cands, key=lambda p: abs(_fnum(p) - didx)))
        if required:
            raise FileNotFoundError(f"No {suffix} frames for {location_id}")
        return None

    def load_image(self, location_id: str, t_key: str) -> Image.Image:
        month = int(t_key[1:])
        didx = self._month_to_didx(location_id, month)
        rgb_path = self._resolve_frame_path(location_id, didx, "rgb", required=True)
        if self.color_mode == "rgb":
            return Image.open(rgb_path).convert("RGB")
        infra_path = self._resolve_frame_path(location_id, didx, "infra", required=False)
        if self.color_mode == "nrg":
            return _compose_nrg(rgb_path, infra_path)
        return _compose_ndvi(rgb_path, infra_path)

    def load_pair_images(self, pair: PairKey) -> Tuple[Image.Image, Image.Image]:
        return (self.load_image(pair.location_id, pair.t1_key),
                self.load_image(pair.location_id, pair.t2_key))

    def load_metadata(self) -> pd.DataFrame:
        rows = []
        base = pd.Timestamp("2018-01-01")
        for pair in self._pairs:
            for t_key in (pair.t1_key, pair.t2_key):
                m = int(t_key[1:])
                rows.append({
                    "location": pair.location_id,
                    "timestamp": base + pd.Timedelta(days=30 * m),
                    "t_key": t_key,
                    "pair_id": f"{pair.location_id}_{pair.t1_key}_{pair.t2_key}",
                    "dataset_name": self.name,
                })
        df = pd.DataFrame(rows).sort_values(["location", "timestamp"])
        return df.reset_index(drop=True)

    def _labels(self, aoi: str) -> np.ndarray:
        if aoi not in self._label_cache:
            self._label_cache[aoi] = np.load(self._labels_dir / f"{aoi}.npy",
                                             allow_pickle=False)
        return self._label_cache[aoi]

    def get_pair_label(self, pair: PairKey) -> Optional[PairLabel]:
        L = self._labels(pair.location_id)
        m1, m2 = int(pair.t1_key[1:]), int(pair.t2_key[1:])
        if m1 >= L.shape[0] or m2 >= L.shape[0]:
            return None
        # The preprocessed labels store class indices 0..6 in the DEN order (0 impervious surface ... 6 snow
        # and ice) with no nodata value; the shared palette reserves 0 for nodata and numbers the classes
        # 1..7, so shift by one. (Checked against the imagery: index 5 has the lowest near-infrared
        # reflectance, index 6 is the brightest and occurs only on two high-mountain AOIs.)
        return derive_pair_label(L[m1].astype(np.int16) + 1, L[m2].astype(np.int16) + 1, self.stable_threshold)

    # -- helpers --------------------------------------------------------
    def text_caption_for_pair(self, pair: PairKey) -> str:
        lb = self.get_pair_label(pair)
        if lb is None:
            return "unknown land-cover change"
        if lb.stable:
            dom = (lb.dominant_t1_class or "unknown").replace("_", " ")
            return f"stable {dom} land cover"
        t1 = (lb.dominant_t1_class or "unknown").replace("_", " ")
        t2 = (lb.dominant_t2_class or "unknown").replace("_", " ")
        return f"{t1} replaced by {t2}"

    def _build_pairs(self) -> List[PairKey]:
        pairs: List[PairKey] = []
        for aoi in self._aois:
            for a, b in zip(self._months[:-1], self._months[1:]):
                pairs.append(PairKey(aoi, f"m{a:02d}", f"m{b:02d}"))
        return pairs
