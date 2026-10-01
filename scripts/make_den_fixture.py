"""
Generate a tiny synthetic Dynamic EarthNet tree for fast end-to-end tests.

Mirrors the real on-disk layout expected by ``src.datasets.dynamic_earthnet``:

    <dest>/
    ├── planet/<aoi>/<YYYY-MM-01>.tif   (4-band RGBNIR uint8)
    ├── labels/<aoi>/<YYYY-MM-01>.tif   (1-band uint8 class indices 0..7)
    ├── labels_index.parquet            (built via download_den.build_label_index)
    └── _done.marker

The fixture is deterministic and engineered so that, under the default
``bimonthly`` pairing, the derived ``PairLabel``s contain a mix of:
  - stable pairs (hard negatives),
  - a real urban-growth transition  (agriculture -> impervious_surface),
  - a deforestation transition      (forest -> soil),
  - a seasonal transition           (snow_and_ice -> forest, i.e. snow-melt).

Usage:
    python -m scripts.make_den_fixture [--dest tests/fixtures/den_tiny] [--force]
"""
from __future__ import annotations

import argparse
import shutil
import zlib
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

# class index -> RGB palette (so the frozen VLM gets weak but non-zero signal)
from src.datasets._palette import DEN_PALETTE_BY_INDEX as _PALETTE

_TILE = 64  # H = W

# Per-AOI, per-month label layout. Each entry is a list of
# (class_index, row_fraction) painted top-down so fractions sum to 1.0.
# Months are 2018-01 .. 2018-08 (index 0..7). Bimonthly pairing picks
# months 0,2,4,6 -> Jan, Mar, May, Jul -> 3 consecutive pairs per AOI.
_LAYOUTS = {
    "1311": {  # urban growth
        "2018-01-01": [(2, 1.0)],                 # agriculture
        "2018-03-01": [(2, 1.0)],                 # agriculture   (Jan->Mar stable)
        "2018-05-01": [(1, 0.6), (2, 0.4)],       # impervious dominant (Mar->May: agri->impervious)
        "2018-07-01": [(1, 0.8), (2, 0.2)],       # impervious     (May->Jul stable impervious)
        # filler months (not selected by bimonthly, kept for realism)
        "2018-02-01": [(2, 1.0)],
        "2018-04-01": [(2, 1.0)],
        "2018-06-01": [(1, 0.7), (2, 0.3)],
        "2018-08-01": [(1, 0.85), (2, 0.15)],
    },
    "2065": {  # snow-melt (seasonal) + deforestation (permanent)
        "2018-01-01": [(7, 0.7), (3, 0.3)],       # winter snow over forest
        "2018-03-01": [(3, 1.0)],                 # forest      (Jan->Mar: snow->forest = SEASONAL)
        "2018-05-01": [(3, 1.0)],                 # forest      (Mar->May stable forest)
        "2018-07-01": [(5, 0.6), (3, 0.4)],       # soil dominant (May->Jul: forest->soil = deforestation)
        "2018-02-01": [(7, 0.5), (3, 0.5)],
        "2018-04-01": [(3, 1.0)],
        "2018-06-01": [(3, 1.0)],
        "2018-08-01": [(5, 0.7), (3, 0.3)],
    },
}


def _label_array(layout, rng) -> np.ndarray:
    arr = np.zeros((_TILE, _TILE), dtype=np.uint8)
    row = 0
    for cls, frac in layout:
        h = int(round(frac * _TILE))
        arr[row:row + h, :] = cls
        row += h
    if row < _TILE:                      # rounding remainder -> last class
        arr[row:, :] = layout[-1][0]
    return arr


def _planet_from_label(label: np.ndarray, rng) -> np.ndarray:
    """Colorize a label map into a 4-band (RGB+NIR) uint8 array [C,H,W]."""
    h, w = label.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    for cls, color in _PALETTE.items():
        rgb[label == cls] = color
    rgb += rng.normal(0, 8, rgb.shape).astype(np.float32)   # mild texture
    rgb = np.clip(rgb, 0, 255)
    nir = np.clip(rgb.mean(axis=2, keepdims=True) * 0.8, 0, 255)
    bands = np.concatenate([rgb, nir], axis=2)              # [H,W,4]
    # uint8 so the DEN loader's uint16 min-max stretch path is skipped and
    # palette colours are preserved exactly (deterministic fixtures).
    return bands.transpose(2, 0, 1).astype(np.uint8)        # [4,H,W]


def _write_tif(path: Path, data: np.ndarray, count: int, dtype: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    transform = from_origin(0, 0, 1, 1)
    with rasterio.open(
        path, "w", driver="GTiff",
        height=data.shape[-2], width=data.shape[-1],
        count=count, dtype=dtype, transform=transform,
    ) as dst:
        if count == 1:
            dst.write(data, 1)
        else:
            dst.write(data)


def build_fixture(dest: Path, force: bool = False) -> Path:
    marker = dest / "_done.marker"
    if marker.exists() and not force:
        print(f"Fixture already present at {dest} (use --force to rebuild).")
        return dest
    if dest.exists() and force:
        shutil.rmtree(dest)

    for aoi, months in _LAYOUTS.items():
        for date_key, layout in sorted(months.items()):
            # A stable digest, not hash(): Python salts str hashes per process, which
            # made the "deterministic" fixture differ between runs.
            seed = zlib.crc32(f"{aoi}|{date_key}".encode())
            rng = np.random.default_rng(seed)
            label = _label_array(layout, rng)
            planet = _planet_from_label(label, rng)
            _write_tif(dest / "planet" / aoi / f"{date_key}.tif", planet, 4, "uint8")
            _write_tif(dest / "labels" / aoi / f"{date_key}.tif", label, 1, "uint8")

    # Build labels_index.parquet using the project's own routine.
    from scripts.download_den import build_label_index
    build_label_index(dest, pairing_strategy="bimonthly")

    marker.touch()
    print(f"\nSynthetic DEN fixture ready at: {dest}")
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build tiny synthetic DEN fixture")
    parser.add_argument("--dest", default="tests/fixtures/den_tiny")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    build_fixture(Path(args.dest), force=args.force)


if __name__ == "__main__":
    main()
