"""
Dynamic EarthNet — native Planet-Fusion 3 m raster loader (official TUM zips).

This is the full-resolution counterpart to :mod:`dynamic_earthnet` (the 75-AOI
preprocessed gdown subset) and :mod:`dynamic_earthnet_pp` (the DynNet ``.npy``
subset). It reads the *official* DynamicEarthNet release (TUM mediatum,
https://mediatum.ub.tum.de/1738088) straight out of its zip archives — nothing is
ever extracted to disk — so the native 3 m Planet surface-reflectance rasters and
the published 7-class labels can be used as a retrieval corpus through the same
``TemporalDataset`` protocol as every other dataset.

Expected on-disk layout (a directory of the downloaded zips)::

    <root>/
    ├── labels.zip          labels/<cube>_<zone>/Labels/Raster/<scene>/<scene>-YYYY_MM_01.tif
    │                       (one-hot [1024,1024,7] uint8 monthly LULC)
    └── planet.<zone>.zip   planet/<zone>/<lonlat>/<cube>/PF-SR/<YYYY-MM-DD>.tif
                            (daily Planet-Fusion surface reflectance, [1024,1024,4] int16,
                             band order Blue/Green/Red/NIR)

Only the first-of-month PF-SR tile that lines up with each monthly label is used,
giving one snapshot per month per cube (``location_id`` = the Planet cube id, e.g.
``1700_3100_13``; ``t_key`` = ``"YYYY-MM"``). The 7-channel one-hot labels are
collapsed to DEN's single-band class-index convention (0 = nodata, 1..7 = the
classes in :data:`_palette.DEN_CLASS_NAMES`) and fed through the shared
:func:`dynamic_earthnet.derive_pair_label`, so the resulting :class:`PairLabel`
objects are identical in schema and class vocabulary to the preprocessed loaders.

Reads only; no zip is mutated and no tile is written to disk. Decoding uses
``rasterio`` (already a project dependency) via an in-memory ``MemoryFile``.

The cube ids carry the UTM zone (``..._<zone>``) so the raw zone archives can be
cherry-picked over rsync (see ``scripts/download_den_planet.py`` / the report's
data section) instead of mirroring the full ~525 GB release.
"""
from __future__ import annotations

import os
import re
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

from ._palette import DEN_CLASS_NAMES
from .base import PairKey, PairLabel
from .dynamic_earthnet import derive_pair_label

# Planet PF-SR band order is [Blue, Green, Red, NIR]. CLIP takes 3 channels, so each
# mode is a 3-band composite mapped to (R, G, B). 0-indexed: B=0, G=1, R=2, NIR=3.
PLANET_BANDS: Dict[str, Tuple[int, int, int]] = {
    "rgb": (2, 1, 0),   # true colour
    "nir": (3, 2, 1),   # NIR-Red-Green false colour (== "nrg" elsewhere)
}
STRETCH_PCT = (2.0, 98.0)  # per-channel percentile stretch -> natural-looking RGB for CLIP

# label folder = "<cube>_<zone>" or "<cube>-<zone>" (e.g. 1700_3100_13_13N, 2235_3403_13-17N)
_CUBE_RE = re.compile(r"^(?P<cube>.+_\d+)[_-](?P<zone>\d+[NS])$")


def planet_to_rgb(arr: np.ndarray, bands: str = "rgb") -> np.ndarray:
    """``[H, W, 4]`` int16 Planet reflectance -> ``[H, W, 3]`` uint8 composite for CLIP.

    Per-channel 2--98 percentile stretch (Planet SR values vary widely per scene),
    which yields a more natural image for the RGB-pretrained CLIP tower than a fixed
    divisor.
    """
    if bands not in PLANET_BANDS:
        raise ValueError(f"Unknown bands {bands!r} (choices: {sorted(PLANET_BANDS)})")
    comp = arr[..., list(PLANET_BANDS[bands])].astype(np.float32)
    out = np.empty(comp.shape, dtype=np.uint8)
    for c in range(3):
        ch = comp[..., c]
        lo, hi = np.percentile(ch, STRETCH_PCT)
        if hi <= lo:
            hi = lo + 1.0
        out[..., c] = (np.clip((ch - lo) / (hi - lo), 0.0, 1.0) * 255).astype(np.uint8)
    return out


def onehot_to_classmap(onehot: np.ndarray) -> np.ndarray:
    """``[H, W, 7]`` uint8 one-hot LULC -> ``[H, W]`` uint8 class-index map.

    Output uses DEN's convention: 0 = nodata (no channel set), 1..7 = the classes in
    :data:`_palette.DEN_CLASS_NAMES` (channel ``i`` -> class index ``i + 1``), matching
    the single-band rasters the preprocessed loaders feed to ``derive_pair_label``.
    """
    has_label = onehot.any(axis=-1)
    classmap = (onehot.argmax(axis=-1) + 1).astype(np.uint8)
    classmap[~has_label] = 0
    return classmap


def _read_tif_from_zip(zf: zipfile.ZipFile, member: str) -> np.ndarray:
    """Decode one GeoTIFF straight from an open zip into ``[H, W, C]`` (no extraction)."""
    import rasterio
    from rasterio.io import MemoryFile

    with MemoryFile(zf.read(member)) as mf:
        with mf.open() as src:
            arr = src.read()           # [C, H, W]
    return np.transpose(arr, (1, 2, 0))    # [H, W, C]


def build_index(root: str) -> dict:
    """Discover usable ``(cube, month)`` pairs = cubes with BOTH a label and Planet imagery.

    Returns ``{zone: {cube: {"label": {month: member}, "planet": {month: member}}}}``,
    keeping only months present on both sides. Missing/corrupt planet zips are skipped.
    """
    lz = zipfile.ZipFile(os.path.join(root, "labels.zip"))
    label_index: Dict[str, Dict[str, str]] = {}
    cube_zone: Dict[str, Tuple[str, str]] = {}
    for name in lz.namelist():
        if "/Labels/Raster/" not in name or not name.endswith(".tif"):
            continue
        cubefolder = name.split("/")[1]
        m = _CUBE_RE.match(cubefolder)
        # label date separator is inconsistent across cubes: 2018_01_01 OR 2018-01-01
        dm = re.search(r"(\d{4})[_-](\d{2})[_-]01\.tif$", name)
        if not m or not dm:
            continue
        month = f"{dm.group(1)}-{dm.group(2)}"
        label_index.setdefault(cubefolder, {})[month] = name
        cube_zone[cubefolder] = (m.group("cube"), m.group("zone"))

    by_zone: Dict[str, Dict[str, Dict[str, str]]] = {}
    for cubefolder, (cube, zone) in cube_zone.items():
        by_zone.setdefault(zone, {})[cube] = label_index[cubefolder]

    index: dict = {}
    for zone, cubes in sorted(by_zone.items()):
        zip_path = os.path.join(root, f"planet.{zone}.zip")
        if not os.path.exists(zip_path):
            print(f"  zone {zone}: no planet.{zone}.zip — skipping {len(cubes)} cube(s)")
            continue
        try:
            pz = zipfile.ZipFile(zip_path)
            members = pz.namelist()
        except zipfile.BadZipFile:
            print(f"  zone {zone}: planet.{zone}.zip is CORRUPT — skipping {len(cubes)} cube(s)")
            continue
        planet_by_cube: Dict[str, Dict[str, str]] = {}
        for n in members:
            if "/PF-SR/" not in n or not n.endswith(".tif"):
                continue
            dm = re.search(r"/([0-9_]+)/PF-SR/(\d{4})-(\d{2})-01\.tif$", n)  # first-of-month only
            if not dm:
                continue
            planet_by_cube.setdefault(dm.group(1), {})[f"{dm.group(2)}-{dm.group(3)}"] = n
        for cube, label_months in cubes.items():
            planet_months = planet_by_cube.get(cube, {})
            shared = sorted(set(label_months) & set(planet_months))
            if len(shared) < 2:    # need >= 2 months to form any change pair
                continue
            index.setdefault(zone, {})[cube] = {
                "label": {mo: label_months[mo] for mo in shared},
                "planet": {mo: planet_months[mo] for mo in shared},
            }
    return index


def _select_months(months: List[str], strategy: str) -> List[str]:
    """Sub-sample sorted ``YYYY-MM`` months per pairing strategy."""
    months = sorted(months)
    if strategy == "monthly":
        return months
    if strategy == "bimonthly":
        return months[::2]
    if strategy == "seasonal-quartet":
        season = {
            "01": "winter", "02": "winter", "12": "winter",
            "03": "spring", "04": "spring", "05": "spring",
            "06": "summer", "07": "summer", "08": "summer",
            "09": "autumn", "10": "autumn", "11": "autumn",
        }
        seen: Dict[str, str] = {}
        for mo in months:
            s = season[mo[5:7]]
            seen.setdefault(s, mo)
        return [seen[s] for s in ["winter", "spring", "summer", "autumn"] if s in seen]
    raise ValueError(f"Unknown pairing_strategy: {strategy!r}")


class DENPlanetDataset:
    """Native 3 m Planet-Fusion DynamicEarthNet — implements ``TemporalDataset``."""

    CLASS_NAMES = DEN_CLASS_NAMES
    name = "dynamic_earthnet_planet"
    temporal_axis_type = "daily"

    def __init__(
        self,
        root: str | Path,
        pairing_strategy: str = "bimonthly",
        bands: str = "rgb",
        stable_threshold: float = 0.02,
        split: Optional[str] = None,
        aoi_filter: Optional[List[str]] = None,
        test_fraction: float = 0.2,
        val_fraction: float = 0.2,
        split_seed: int = 42,
    ) -> None:
        self.root = str(root)
        self.pairing_strategy = pairing_strategy
        self.bands = bands
        self.stable_threshold = stable_threshold
        self.split = split

        # The official release has no published train/val/test split, so we derive a
        # deterministic AOI-level partition (disjoint cube sets, seeded) — the same
        # cube-holdout idea as the report's cross-validation, just a single fixed fold
        # here so ``run_pipeline`` can train on one split and evaluate on another
        # without leakage. ``split=None`` / ``"all"`` returns the whole corpus.
        full_index = build_index(self.root)
        all_cubes = sorted(c for cubes in full_index.values() for c in cubes)
        selected = set(self._select_split(
            all_cubes, split, test_fraction, val_fraction, split_seed))
        if aoi_filter:
            selected &= set(aoi_filter)

        # Flatten build_index into (cube, month) -> member lookups + per-cube month lists.
        self._cube_zone: Dict[str, str] = {}
        self._planet_member: Dict[Tuple[str, str], str] = {}
        self._label_member: Dict[Tuple[str, str], str] = {}
        self._cube_months: Dict[str, List[str]] = {}
        for zone, cubes in full_index.items():
            for cube, members in cubes.items():
                if cube not in selected:
                    continue
                self._cube_zone[cube] = zone
                self._cube_months[cube] = sorted(members["planet"])
                for month, member in members["planet"].items():
                    self._planet_member[(cube, month)] = member
                for month, member in members["label"].items():
                    self._label_member[(cube, month)] = member

        self._zips: Dict[str, zipfile.ZipFile] = {}   # cached open zip handles, keyed by archive
        self._pairs: List[PairKey] = self._build_pairs()
        self._label_cache: Dict[Tuple[str, str, str], PairLabel] = {}

    # ------------------------------------------------------------------
    # TemporalDataset protocol
    # ------------------------------------------------------------------

    def list_locations(self) -> List[str]:
        return sorted(self._cube_months.keys())

    def list_pairs(self) -> List[PairKey]:
        return list(self._pairs)

    def load_image(self, location_id: str, t_key: str) -> Image.Image:
        member = self._planet_member.get((location_id, t_key))
        if member is None:
            raise FileNotFoundError(f"No Planet tile for cube {location_id!r} month {t_key!r}")
        zf = self._zip_for(f"planet.{self._cube_zone[location_id]}.zip")
        arr = _read_tif_from_zip(zf, member)               # [1024, 1024, 4] int16
        return Image.fromarray(planet_to_rgb(arr, self.bands))

    def load_pair_images(self, pair: PairKey) -> Tuple[Image.Image, Image.Image]:
        return (
            self.load_image(pair.location_id, pair.t1_key),
            self.load_image(pair.location_id, pair.t2_key),
        )

    def load_metadata(self) -> pd.DataFrame:
        rows = []
        for pair in self._pairs:
            for t_pos, t_key in enumerate([pair.t1_key, pair.t2_key]):
                rows.append({
                    "location": pair.location_id,
                    "timestamp": pd.Timestamp(t_key),     # "YYYY-MM" -> first of month
                    "t_key": t_key,
                    "pair_id": f"{pair.location_id}_{pair.t1_key}_{pair.t2_key}",
                    "dataset_name": self.name,
                    "timepoint_idx": t_pos,
                })
        df = pd.DataFrame(rows)
        df.sort_values(["location", "timestamp"], inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def get_pair_label(self, pair: PairKey) -> Optional[PairLabel]:
        cache_key = (pair.location_id, pair.t1_key, pair.t2_key)
        if cache_key in self._label_cache:
            return self._label_cache[cache_key]
        try:
            arr_t1 = self._load_classmap(pair.location_id, pair.t1_key)
            arr_t2 = self._load_classmap(pair.location_id, pair.t2_key)
        except FileNotFoundError:
            return None
        label = derive_pair_label(arr_t1, arr_t2, self.stable_threshold)
        self._label_cache[cache_key] = label
        return label

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _zip_for(self, archive_name: str) -> zipfile.ZipFile:
        zf = self._zips.get(archive_name)
        if zf is None:
            zf = zipfile.ZipFile(os.path.join(self.root, archive_name))
            self._zips[archive_name] = zf
        return zf

    def _load_classmap(self, cube: str, month: str) -> np.ndarray:
        member = self._label_member.get((cube, month))
        if member is None:
            raise FileNotFoundError(f"No label for cube {cube!r} month {month!r}")
        onehot = _read_tif_from_zip(self._zip_for("labels.zip"), member)  # [1024, 1024, 7] uint8
        return onehot_to_classmap(onehot)

    def _build_pairs(self) -> List[PairKey]:
        pairs: List[PairKey] = []
        for cube, months in sorted(self._cube_months.items()):
            reps = _select_months(months, self.pairing_strategy)
            for i in range(len(reps) - 1):
                pairs.append(PairKey(location_id=cube, t1_key=reps[i], t2_key=reps[i + 1]))
        return pairs

    @staticmethod
    def _select_split(
        all_cubes: List[str],
        split: Optional[str],
        test_fraction: float,
        val_fraction: float,
        seed: int,
    ) -> List[str]:
        """Deterministic, disjoint AOI partition. ``None``/``"all"`` -> every cube."""
        if not split or split == "all":
            return list(all_cubes)
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be train|val|test|all|None, got {split!r}")
        order = np.random.default_rng(seed).permutation(len(all_cubes))
        shuffled = [all_cubes[i] for i in order]
        n = len(shuffled)
        n_test = round(test_fraction * n)
        n_val = round(val_fraction * n)
        buckets = {
            "test": shuffled[:n_test],
            "val": shuffled[n_test:n_test + n_val],
            "train": shuffled[n_test + n_val:],
        }
        return sorted(buckets[split])
