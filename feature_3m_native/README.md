# Native 3 m Planet-Fusion DEN

Change-retrieval evaluation run **directly on the native 3 m Planet-Fusion
imagery** of DynamicEarthNet — the source that the report's §Data had **excluded**
in favour of the ~7 GB preprocessed RGB/NIR-JPEG subset.

**Question:** does working on the lossy JPEG subset *cost* retrieval performance?
**Answer: no** — the native rasters land in the same band, no worse within fold
variance, so the data-source choice is vindicated.

## Motivation — the excluded source

All DEN results in the report run on either the preprocessed RGB/NIR-**JPEG**
subset (~7 GB, lossy) or downsampled composites. The report's §Data deliberately
**excluded** the full ~525 GB raw TUM mirror. That left one question open: *are we
losing accuracy because we evaluate on compressed / resampled imagery rather than
the full-quality source?*

Answering it requires evaluating on the **native Planet-Fusion surface-reflectance
rasters** — full radiometric depth, no JPEG loss.

## What was added (and where)

**Architecture integration** (additive — does not change any existing dataset):
- `src/datasets/dynamic_earthnet_planet.py` — the `DENPlanetDataset` loader. Reads
  the PF-SR `int16` files (1024×1024×**4**, Blue/Green/Red/NIR, 3 m) **straight from
  the compressed `planet.<UTM>.zip` archives** via `rasterio` MemoryFile (no lossy
  re-encoding, no need to unpack 525 GB), converts the 7-channel one-hot masks to a
  class-index map and reuses DEN's `derive_pair_label`.
- `src/datasets/registry.py`, `src/queries/den.py` — two registration lines
  (`dynamic_earthnet_planet`). This is the registry's **intended extension point**;
  adding a dataset name cannot break the existing loaders.
- `scripts/download_den_planet.py` — `rsync` cherry-pick of **only** the needed UTM
  zones + `labels.zip` (instead of the full mirror).

**Evaluation** (self-contained — does **not** touch `scripts/cv_eval.py`):
- `feature_3m_native/cv_eval_planet3m.py` — 5-fold AOI cross-validation (zero-shot +
  leakage-free PEFT), plus a full-corpus AP estimate with bootstrap CIs and
  permutation p-values. It mirrors the logic of `scripts/cv_eval.py` so the numbers
  are **directly comparable**, while importing the `src/` library read-only.
- `feature_3m_native/jpeg_ablation.py` — **controlled compression/resolution
  ablation**: holds AOIs, encoder, colour and folds fixed and varies only the per-tile
  degradation (JPEG quality {95,75,50,25,10}; downsampling {512,256,128,64}px + JPEG
  q75). Confirms image fidelity is not the bottleneck — the largest Δ from native across the
  JPEG-quality sweep is 0.036 mAP (within one fold std); the resolution sweep reaches $+0.086$ mAP at
  64px, but within that setting's inflated fold variance (std up to 0.21, few positives per small
  fold), so no degradation is detectable at this corpus size — same null under GeoRSCLIP. Plot:
  `scripts/make_jpeg_ablation_figure.py`.
- `feature_3m_native/temporal_pinpoint.py` — **temporal pinpointing**: ranks the
  monthly steps within each AOI timeline (*when* does the change occur) by zero-shot
  Δ-similarity, with a within-AOI permutation test + BH-FDR. CLIP macro temporal mAP
  0.309 vs 0.207 floor, ±1-month peak-hit 63%; encoder-dependent (GeoRSCLIP 0.146, at
  or below chance). Plot: `scripts/make_temporal_pinpoint_figure.py`.
- `feature_3m_native/results/` — the output JSON for all three experiments.

## How to run

```bash
# 1) (once) encode every cube into a single cache — needs the disk holding the zips
uv run python -m src.embeddings --dataset dynamic_earthnet_planet \
  --root /path/to/dir/with/planet.<UTM>.zip --split all --color-mode rgb

# 2) cross-validation (zero-shot + PEFT)
uv run python feature_3m_native/cv_eval_planet3m.py \
  --root /path/to/dir/with/planet.<UTM>.zip --folds 5 --peft

# 3) compression/resolution ablation (+ figure)
uv run python feature_3m_native/jpeg_ablation.py \
  --root /path/to/dir/with/planet.<UTM>.zip --folds 5
uv run python scripts/make_jpeg_ablation_figure.py

# 4) temporal pinpointing (+ figure)
uv run python feature_3m_native/temporal_pinpoint.py \
  --root /path/to/dir/with/planet.<UTM>.zip
uv run python scripts/make_temporal_pinpoint_figure.py
```

## Result (CLIP ViT-L/14, 23 AOIs, 253 pairs, 5-fold AOI CV, dominant relevance)

| Source | colour | macro mAP (zero-shot) | corpus |
|--------|--------|------------------------|--------|
| **Native 3 m raster** | RGB | **0.130 ± 0.068** | 23 AOIs |
| JPEG subset (committed) | NRG | 0.076 | 75 AOIs |
| JPEG subset (fraction relevance) | NRG | 0.123 | 75 AOIs |

The JPEG-subset rows use the labels of the preprocessed loader, whose class names were one class off
(see the notes on the report in the repository README); the native row uses the native loader.

The comparison is not fully controlled (the corpora differ in AOI count, 23 vs 75,
and colour composite, RGB vs NRG). But colour barely moves the score on this corpus
(georsclip on the JPEG subset: RGB 0.085 vs NRG 0.100), so the direction is robust:
**the native source is no worse.**

> The leakage-free k-fold PEFT estimate is 0.215 ± 0.306, but it is **unstable** (one
> fold at 0.75, the rest 0.03–0.19, driven by few positives per small fold), so the
> zero-shot figure is the reliable point of comparison. Only **5/10** queries are
> evaluable — the others have no positives across the 23 AOIs, the same
> data-coverage limit the report notes for the JPEG corpus.

## How this connects to the wider conclusions

This is one more structurally different attempt at the same ceiling. The report
shows that NIR false colour and patch-level scoring help (Sections 8.1 and 8.3) and
that the retrieval ceiling stays near 0.20; the native 3 m result adds that **full
radiometric quality at native resolution does not move it either.** So the limit is
not imagery compression or resolution; it is the frozen web-pretrained features and
the small labelled AOI set. Practical takeaway for the report: the lightweight ~7 GB
JPEG subset cost no accuracy.

## Pointers
- Shared report: [`report/main.pdf`](../report/main.pdf) (source `report/main.tex`), appendix `sec:nativeraster` — tables
  `tab:nativeraster` (adapter, 5-seed stratified AOI hold-out — external one-off, not
  reproducible from the tree) and `tab:native3mcv` (this repository's own reproducible
  dominant-class `cv_eval` benchmark), plus the compression/resolution ablation
  (`tab:jpegablation`, `tab:jpegablationgeors`, `fig:jpegablation`).
- Temporal pinpointing: [`report/main.pdf`](../report/main.pdf), appendix `sec:temporalpinpoint`
  (`tab:temporalpinpoint`, `fig:temporalpinpoint`).
- Loader / integration: `src/datasets/dynamic_earthnet_planet.py`.
