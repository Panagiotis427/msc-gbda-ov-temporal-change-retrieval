"""
Single source of truth for Dynamic EarthNet class names and the synthetic-
fixture RGB palette. Imported by the DEN loader, the fixture generator, and
the test mocks so they cannot drift.

Index 0 = nodata. Indices 1..7 are the 7 LULC classes used by DEN
(Toker et al. 2022). The preprocessed ``labels/<AOI>.npy`` rasters store the same
classes as 0..6 with no nodata value; ``DENNpyDataset`` adds one on reading.
"""
from __future__ import annotations

from typing import Dict, Tuple

DEN_CLASS_NAMES: Tuple[str, ...] = (
    "nodata",
    "impervious_surface",
    "agriculture",
    "forest_and_other_vegetation",
    "wetlands",
    "soil",
    "water",
    "snow_and_ice",
)

DEN_PALETTE: Dict[str, Tuple[int, int, int]] = {
    "nodata": (0, 0, 0),
    "impervious_surface": (128, 128, 128),
    "agriculture": (170, 170, 50),
    "forest_and_other_vegetation": (20, 100, 20),
    "wetlands": (40, 120, 120),
    "soil": (140, 90, 40),
    "water": (20, 40, 160),
    "snow_and_ice": (235, 235, 245),
}

# Same palette keyed by class index (0..7) for code that paints from uint8
# label arrays (the fixture generator).
DEN_PALETTE_BY_INDEX: Dict[int, Tuple[int, int, int]] = {
    i: DEN_PALETTE[c] for i, c in enumerate(DEN_CLASS_NAMES)
}
