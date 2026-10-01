"""
Value-level tests for the DEN preprocessed colour-mode composition
(`_compose_nrg`, `_compose_ndvi`).

The report §8.3 NRG/NDVI mAP numbers depend on this pixel math, which was previously
untested. These exercise the band ordering and the NDVI arithmetic directly on
tiny *lossless* (PNG) inputs — no dataset download, no GPU, pure numpy/PIL —
so they belong in the fast suite.

Inputs are chosen so the NDVI mapping lands on exact uint8 values (the only
non-trivial one is 191; verified stable in both float32 and float64).
"""
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.datasets.dynamic_earthnet_pp import _compose_ndvi, _compose_nrg


def _save_png(arr: np.ndarray, path: Path) -> Path:
    Image.fromarray(arr).save(path)
    return path


# 2x2 RGB + NIR. Per-pixel (R, NIR) picked for clean NDVI:
#   (0,0) R=64  NIR=192 -> ndvi=+0.5 -> (1.5)*127.5 = 191.25 -> 191
#   (0,1) R=200 NIR=0   -> ndvi=-1.0 -> 0
#   (1,0) R=0   NIR=0   -> ndvi= 0   -> 127.5 -> 127
#   (1,1) R=50  NIR=50  -> ndvi= 0   -> 127
_RGB = np.array(
    [[[64, 10, 20], [200, 30, 40]],
     [[0, 50, 60], [50, 70, 80]]],
    dtype=np.uint8,
)
_NIR = np.array([[192, 0], [0, 50]], dtype=np.uint8)


def test_compose_nrg_band_order(tmp_path):
    """NRG must be [NIR, Red, Green] per pixel (vegetation false colour)."""
    rgb_p = _save_png(_RGB, tmp_path / "a_rgb.png")
    nir_p = _save_png(_NIR, tmp_path / "a_infra.png")

    out = np.array(_compose_nrg(rgb_p, nir_p))

    assert out.shape == (2, 2, 3)
    assert out.dtype == np.uint8
    np.testing.assert_array_equal(out[:, :, 0], _NIR)           # channel 0 = NIR
    np.testing.assert_array_equal(out[:, :, 1], _RGB[:, :, 0])  # channel 1 = Red
    np.testing.assert_array_equal(out[:, :, 2], _RGB[:, :, 1])  # channel 2 = Green


def test_compose_nrg_missing_infra_falls_back_to_red(tmp_path):
    """With no NIR frame, NIR <- Red, so channel 0 == channel 1 == Red."""
    rgb_p = _save_png(_RGB, tmp_path / "b_rgb.png")
    missing = tmp_path / "b_infra.png"  # deliberately not created

    with pytest.warns(RuntimeWarning, match="NIR frame missing"):
        out = np.array(_compose_nrg(rgb_p, missing))

    np.testing.assert_array_equal(out[:, :, 0], _RGB[:, :, 0])
    np.testing.assert_array_equal(out[:, :, 1], _RGB[:, :, 0])
    np.testing.assert_array_equal(out[:, :, 2], _RGB[:, :, 1])


def test_compose_ndvi_values(tmp_path):
    """NDVI = (NIR-R)/(NIR+R+eps) -> (ndvi+1)*127.5 -> uint8, replicated x3."""
    rgb_p = _save_png(_RGB, tmp_path / "c_rgb.png")
    nir_p = _save_png(_NIR, tmp_path / "c_infra.png")

    out = np.array(_compose_ndvi(rgb_p, nir_p))

    expected = np.array([[191, 0], [127, 127]], dtype=np.uint8)
    assert out.shape == (2, 2, 3)
    for c in range(3):  # single NDVI band replicated to 3 channels
        np.testing.assert_array_equal(out[:, :, c], expected)


def test_compose_ndvi_missing_infra_is_neutral(tmp_path):
    """No NIR -> nir=red -> ndvi=0 everywhere -> uint8 127 (the +eps guard avoids 0/0)."""
    rgb_p = _save_png(_RGB, tmp_path / "d_rgb.png")
    missing = tmp_path / "d_infra.png"

    with pytest.warns(RuntimeWarning, match="NIR frame missing"):
        out = np.array(_compose_ndvi(rgb_p, missing))
    np.testing.assert_array_equal(out, np.full((2, 2, 3), 127, dtype=np.uint8))


def test_npy_labels_are_read_as_den_indices_0_to_6():
    """The preprocessed label rasters store the seven DEN classes as 0..6 with no nodata value, so index 0 is
    impervious surface and index 6 snow and ice, never nodata and water."""
    from src.datasets.base import PairKey
    from src.datasets.dynamic_earthnet_pp import DENNpyDataset
    ds = object.__new__(DENNpyDataset)            # no dataset on disk: only the label path is exercised
    ds.stable_threshold = 0.02
    L = np.zeros((3, 4, 4), dtype=np.uint8)       # month 0: impervious surface everywhere (index 0)
    L[2] = 6                                       # month 2: snow and ice everywhere (index 6)
    ds._label_cache = {"aoi": L}
    lb = ds.get_pair_label(PairKey("aoi", "m00", "m02"))
    assert lb.dominant_t1_class == "impervious_surface" and lb.dominant_t2_class == "snow_and_ice"
    assert lb.class_change_mask_fraction["snow_and_ice"]["gained_fraction"] == 1.0
    assert lb.class_change_mask_fraction["impervious_surface"]["lost_fraction"] == 1.0
    assert not lb.stable
