"""
Dynamic EarthNet dataset loader (the preprocessed 75-AOI subset).

Expected on-disk layout (after running ``scripts/download_den.py``):

    <root>/
    ├── planet/<aoi_id>/<YYYY-MM-DD>.tif     (daily Planet-Fusion, 4-band RGBNIR)
    ├── labels/<aoi_id>/<YYYY-MM-01>.tif     (monthly 7-class LULC)
    └── labels_index.parquet                  (built by download_den.py)
        columns: location, t1_key, t2_key, change_type, stable,
        dominant_t1_class, dominant_t2_class, and (current schema)
        ``class_change_fraction_json`` — a JSON string mapping each class name to
        ``{"gained_fraction": float, "lost_fraction": float}``. Indexes built
        before that column existed still load; the loader derives the fractions
        from the rasters on demand as a fallback.

Pairing strategies
------------------
``bimonthly`` (default)
    First-of-month tile per AOI, sub-sampled to every other month
    → ≤11 consecutive pairs per AOI (825 pairs over the 75 AOIs).

``monthly``
    All first-of-month tiles → ≤23 pairs per AOI.

``seasonal-quartet``
    Winter / spring / summer / autumn representative tiles
    → ≤3 pairs per AOI.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

from .base import PairKey, PairLabel


# 7 LULC classes (Toker et al. 2022). Index 0 = nodata. Single source of
# truth in ``_palette.py`` so loader, fixture generator, and tests cannot drift.
from ._palette import DEN_CLASS_NAMES as CLASS_NAMES  # noqa: E402

# Parquet column holding the JSON-encoded per-class change fractions. Single
# source of truth shared by the reader (this module), the writer
# (``scripts/download_den.build_label_index``), and the round-trip test, so the
# name cannot drift between them.
FRAC_JSON_COL = "class_change_fraction_json"


def discover_aoi_dates(planet_dir: Path) -> Dict[str, List[str]]:
    """Return {aoi_id: sorted list of date strings 'YYYY-MM-DD'} found on disk."""
    result: Dict[str, List[str]] = {}
    if not planet_dir.exists():
        return result
    for aoi_path in sorted(planet_dir.iterdir()):
        if not aoi_path.is_dir():
            continue
        dates = sorted(
            p.stem for p in aoi_path.glob("*.tif")
        )
        if dates:
            result[aoi_path.name] = dates
    return result


def _select_representative_dates(
    dates: List[str],
    strategy: str,
) -> List[str]:
    """Select representative dates from a list of 'YYYY-MM-DD' strings."""
    if not dates:
        return []

    by_month: Dict[str, str] = {}
    for d in dates:
        ym = d[:7]   # 'YYYY-MM'
        if ym not in by_month:
            by_month[ym] = d

    sorted_months = sorted(by_month)

    if strategy == "bimonthly":
        return [by_month[m] for m in sorted_months[::2]]   # every other month
    if strategy == "monthly":
        return [by_month[m] for m in sorted_months]
    if strategy == "seasonal-quartet":
        season_map = {
            "01": "winter", "02": "winter",
            "03": "spring", "04": "spring", "05": "spring",
            "06": "summer", "07": "summer", "08": "summer",
            "09": "autumn", "10": "autumn", "11": "autumn",
            "12": "winter",
        }
        seen: Dict[str, str] = {}
        for m in sorted_months:
            s = season_map[m[5:7]]
            if s not in seen:
                seen[s] = by_month[m]
        return [seen[s] for s in ["winter", "spring", "summer", "autumn"] if s in seen]
    raise ValueError(f"Unknown pairing_strategy: {strategy!r}")


def derive_pair_label(
    label_t1: np.ndarray,
    label_t2: np.ndarray,
    stable_threshold: float = 0.02,
) -> PairLabel:
    """Derive a ``PairLabel`` from two monthly LULC rasters (uint8 H×W)."""
    valid = (label_t1 > 0) & (label_t2 > 0)
    n_valid = int(valid.sum())
    if n_valid == 0:
        return PairLabel(
            change_type="unknown",
            stable=True,
            dominant_t1_class=None,
            dominant_t2_class=None,
            class_change_mask_fraction={},
        )

    per_class: Dict[str, Dict[str, float]] = {}
    for c in range(1, 8):
        was_c = (label_t1 == c) & valid
        is_c = (label_t2 == c) & valid
        gained = (~was_c) & is_c
        lost = was_c & (~is_c)
        per_class[CLASS_NAMES[c]] = {
            "gained_fraction": float(gained.sum()) / n_valid,
            "lost_fraction": float(lost.sum()) / n_valid,
        }

    def dominant(arr: np.ndarray) -> str:
        counts = np.bincount(arr[valid].ravel(), minlength=8)
        idx = int(counts[1:].argmax()) + 1
        return CLASS_NAMES[idx]

    dom_t1 = dominant(label_t1)
    dom_t2 = dominant(label_t2)

    total_change = sum(
        v["gained_fraction"] + v["lost_fraction"] for v in per_class.values()
    ) / 2
    stable = total_change < stable_threshold
    change_type = "stable" if (stable or dom_t1 == dom_t2) else f"{dom_t1}->{dom_t2}"

    return PairLabel(
        change_type=change_type,
        stable=stable,
        dominant_t1_class=dom_t1,
        dominant_t2_class=dom_t2,
        class_change_mask_fraction=per_class,
    )


def _load_tif_rgb(path: Path) -> Image.Image:
    """Load a GeoTIFF as an RGB PIL image (drops NIR band if 4-band).

    The first three bands are taken as (R, G, B) and non-uint8 data is stretched by
    each image's own min/max. The real Planet-Fusion band order is B, G, R, NIR (see
    :mod:`dynamic_earthnet_planet`), so real tiles would come out as BGR with a
    per-image stretch; only the synthetic test fixture (uint8, palette colours in
    bands 0-2) goes through this function.
    """
    try:
        import rasterio  # type: ignore
        with rasterio.open(path) as src:
            bands = src.read()   # [C, H, W] uint16 or uint8
        rgb = bands[:3].transpose(1, 2, 0)   # [H, W, 3]
        if rgb.dtype != np.uint8:
            mn, mx = rgb.min(), rgb.max()
            if mx > mn:
                rgb = ((rgb - mn) / (mx - mn) * 255).astype(np.uint8)
            else:
                rgb = np.zeros_like(rgb, dtype=np.uint8)
        return Image.fromarray(rgb)
    except ImportError:
        # Fallback: use PIL (works only for single-band / basic TIFs)
        img = Image.open(path).convert("RGB")
        return img


def _load_label_tif(path: Path) -> np.ndarray:
    """Load a LULC GeoTIFF as a uint8 [H, W] numpy array (class indices)."""
    try:
        import rasterio  # type: ignore
        with rasterio.open(path) as src:
            arr = src.read(1)    # single-band [H, W]
        return arr.astype(np.uint8)
    except ImportError:
        arr = np.array(Image.open(path))
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        return arr.astype(np.uint8)


class DENDataset:
    """Dynamic EarthNet dataset — implements ``TemporalDataset`` protocol."""

    CLASS_NAMES = CLASS_NAMES
    name = "dynamic_earthnet"
    temporal_axis_type = "daily"

    def __init__(
        self,
        root: str | Path,
        pairing_strategy: str = "bimonthly",
        stable_threshold: float = 0.02,
        aoi_filter: Optional[List[str]] = None,
    ) -> None:
        self.root = Path(root)
        self.pairing_strategy = pairing_strategy
        self.stable_threshold = stable_threshold

        self._planet_dir = self.root / "planet"
        self._label_dir = self.root / "labels"
        self._index_path = self.root / "labels_index.parquet"

        # Discover AOIs and dates on disk
        self._aoi_dates = discover_aoi_dates(self._planet_dir)
        if aoi_filter:
            self._aoi_dates = {k: v for k, v in self._aoi_dates.items() if k in aoi_filter}

        self._pairs: List[PairKey] = self._build_pairs()
        self._label_cache: Dict[str, PairLabel] = {}

        # Load pre-built index if available (speeds up label access)
        if self._index_path.exists():
            self._index_df = pd.read_parquet(self._index_path)
            if FRAC_JSON_COL not in self._index_df.columns:
                import warnings
                warnings.warn(
                    f"{self._index_path} predates the '{FRAC_JSON_COL}' column; "
                    "fraction-based relevance queries will derive fractions from "
                    "the rasters per pair (slower, and impossible if the rasters "
                    "are absent). Rebuild with: python -m scripts.download_den "
                    "(or scripts.download_den.build_label_index).",
                    stacklevel=2,
                )
        else:
            self._index_df = None

    # ------------------------------------------------------------------
    # TemporalDataset protocol
    # ------------------------------------------------------------------

    def list_locations(self) -> List[str]:
        return sorted(self._aoi_dates.keys())

    def list_pairs(self) -> List[PairKey]:
        return list(self._pairs)

    def load_image(self, location_id: str, t_key: str) -> Image.Image:
        path = self._planet_dir / location_id / f"{t_key}.tif"
        if not path.exists():
            raise FileNotFoundError(f"Planet tile not found: {path}")
        return _load_tif_rgb(path)

    def load_pair_images(self, pair: PairKey) -> Tuple[Image.Image, Image.Image]:
        return (
            self.load_image(pair.location_id, pair.t1_key),
            self.load_image(pair.location_id, pair.t2_key),
        )

    def load_metadata(self) -> pd.DataFrame:
        rows = []
        for pair_idx, pair in enumerate(self._pairs):
            for t_pos, t_key in enumerate([pair.t1_key, pair.t2_key]):
                rows.append(
                    {
                        "location": pair.location_id,
                        "timestamp": pd.Timestamp(t_key),
                        "t_key": t_key,
                        "pair_id": f"{pair.location_id}_{pair.t1_key}_{pair.t2_key}",
                        "dataset_name": self.name,
                        "timepoint_idx": t_pos,
                    }
                )
        df = pd.DataFrame(rows)
        df.sort_values(["location", "timestamp"], inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def get_pair_label(self, pair: PairKey) -> Optional[PairLabel]:
        cache_key = f"{pair.location_id}__{pair.t1_key}__{pair.t2_key}"
        if cache_key in self._label_cache:
            return self._label_cache[cache_key]

        # Try pre-built index first
        if self._index_df is not None:
            row = self._index_df[
                (self._index_df["location"] == pair.location_id)
                & (self._index_df["t1_key"] == pair.t1_key)
                & (self._index_df["t2_key"] == pair.t2_key)
            ]
            if len(row) == 1:
                r = row.iloc[0]
                # Per-class change fractions back the fraction-based relevance
                # predicates (benchmark._gained / _lost); without them those
                # queries silently match nothing. Prefer the fractions persisted
                # in the index (raster-free, and consistent with the index's
                # scalar fields since the same build_label_index pass wrote both).
                # Older indexes predate that column -> best-effort derive from the
                # rasters as a fallback, kept robust: only fill fractions if the
                # rasters load cleanly (the index's scalar label is always used),
                # since _derive_label catches only FileNotFoundError, not corrupt
                # or dimension-mismatched rasters.
                fractions: Dict[str, Dict[str, float]] = {}
                frac_json = r.get(FRAC_JSON_COL)
                if isinstance(frac_json, str):
                    decoded = json.loads(frac_json)
                    # Guard against a hand-edited/foreign index: downstream
                    # predicates expect a dict-of-dicts, so anything else -> {}.
                    fractions = decoded if isinstance(decoded, dict) else {}
                else:
                    try:
                        derived = self._derive_label(pair)
                        if derived is not None:
                            fractions = derived.class_change_mask_fraction
                    except FileNotFoundError:
                        fractions = {}  # rasters absent — expected, stay quiet
                    except Exception as exc:  # corrupt / mismatched raster
                        import warnings
                        warnings.warn(
                            f"Could not derive change fractions for {pair}: "
                            f"{exc!r}; leaving them empty.",
                            stacklevel=2,
                        )
                        fractions = {}
                label = PairLabel(
                    change_type=str(r["change_type"]),
                    stable=bool(r["stable"]),
                    dominant_t1_class=r.get("dominant_t1_class"),
                    dominant_t2_class=r.get("dominant_t2_class"),
                    class_change_mask_fraction=fractions,
                )
                self._label_cache[cache_key] = label
                return label

        # Derive from LULC rasters
        label = self._derive_label(pair)
        self._label_cache[cache_key] = label
        return label

    # ------------------------------------------------------------------
    # DEN-specific helpers
    # ------------------------------------------------------------------

    def list_stable_pairs(self) -> List[PairKey]:
        return [p for p in self._pairs if self._is_stable(p)]

    def list_transition_pairs(
        self,
        *,
        t1_class: Optional[str] = None,
        t2_class: Optional[str] = None,
    ) -> List[PairKey]:
        result = []
        for p in self._pairs:
            label = self.get_pair_label(p)
            if label is None or label.stable:
                continue
            if t1_class and label.dominant_t1_class != t1_class:
                continue
            if t2_class and label.dominant_t2_class != t2_class:
                continue
            result.append(p)
        return result

    def load_label_array(self, aoi: str, date_key: str) -> np.ndarray:
        """Return LULC label array for *aoi* at *date_key* (nearest monthly label)."""
        month_key = date_key[:7] + "-01"
        path = self._label_dir / aoi / f"{month_key}.tif"
        if not path.exists():
            # Try without day
            candidates = sorted((self._label_dir / aoi).glob("*.tif"))
            # pick nearest by year-month
            target_ym = date_key[:7]
            candidates_ym = [(c, c.stem[:7]) for c in candidates]
            best = min(candidates_ym, key=lambda x: abs(
                pd.Timestamp(x[1]) - pd.Timestamp(target_ym)
            ), default=None)
            if best is None:
                raise FileNotFoundError(f"No label tile near {date_key} for AOI {aoi}")
            path = best[0]
        return _load_label_tif(path)

    def text_caption_for_pair(self, pair: PairKey) -> str:
        label = self.get_pair_label(pair)
        if label is None:
            return "unknown land-cover change"
        if label.stable:
            dom = label.dominant_t1_class or "unknown"
            return f"stable {dom.replace('_', ' ')} land cover"
        t1 = (label.dominant_t1_class or "unknown").replace("_", " ")
        t2 = (label.dominant_t2_class or "unknown").replace("_", " ")
        return f"{t1} replaced by {t2}"

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _build_pairs(self) -> List[PairKey]:
        pairs: List[PairKey] = []
        for aoi, dates in sorted(self._aoi_dates.items()):
            reps = _select_representative_dates(dates, self.pairing_strategy)
            for i in range(len(reps) - 1):
                pairs.append(PairKey(
                    location_id=aoi,
                    t1_key=reps[i],
                    t2_key=reps[i + 1],
                ))
        return pairs

    def _is_stable(self, pair: PairKey) -> bool:
        label = self.get_pair_label(pair)
        return label is None or label.stable

    def _derive_label(self, pair: PairKey) -> Optional[PairLabel]:
        try:
            arr_t1 = self.load_label_array(pair.location_id, pair.t1_key)
            arr_t2 = self.load_label_array(pair.location_id, pair.t2_key)
            return derive_pair_label(arr_t1, arr_t2, self.stable_threshold)
        except FileNotFoundError:
            return None
