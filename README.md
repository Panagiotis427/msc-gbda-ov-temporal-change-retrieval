# Open Vocabulary Temporal Change Retrieval (GBDA Lab Project)

> **Results → [Results at a glance](#results-at-a-glance) below; full report → [`report/main.pdf`](report/main.pdf)** (best: GeoRSCLIP+NRG `patch_top3`, CV mAP 0.193 ± 0.051, 4/9 FDR-significant).

A *Semantic Change Search Engine*: given a natural-language query
(e.g. *"new buildings on former agricultural land"*, *"forest cleared to bare
soil"*), retrieve the satellite image **pairs and the timestep** where that
change occurred — across a multitemporal dataset, without training a
class-specific detector.

Frozen vision-language backbones (CLIP / GeoRSCLIP / RemoteCLIP) encode each
timestep; a bi-temporal *change feature* is matched against the query text.
Primary dataset: **Dynamic EarthNet (DEN)**; the dataset-agnostic registry also
runs **QFabric** (construction change-types), **LEVIR-CC/MCI** (human-captioned
building/road change + masks) and **SECOND-CC** (a six-class land-cover open
vocabulary + semantic masks), with LEVIR-MCI and SECOND-CC masks driving
quantitative change localisation.

> **Just want to run the app?** Jump to [Run / install / use](#run--install--use) —
> install, then a 30-second synthetic demo or the real dataset.

## Demo

**Try it live:** the current UI runs over a bundled synthetic corpus, no install, on the
[deployed HuggingFace Space](https://huggingface.co/spaces/panagiotis427/Open_Vocabulary_Temporal_Change_Retrieval).

![Gradio UI — semantic change search engine](report/figures/app_screenshot.png)

**Screen recordings** (`demos/`, click to play on GitHub) —
[1](demos/demo_1.mp4) · [2](demos/demo_2.mp4) · [3](demos/demo_3.mp4) the app in default settings
(LEVIR-MCI — the documented default, sharing LEVIR-CC's pairs — GeoRSCLIP, zero-shot) running the built-in example searches;
[4](demos/demo_4.mp4) a custom free-text query (open-vocabulary);
[5](demos/demo_5.mp4) switching dataset (LEVIR-MCI → Dynamic EarthNet) and scoring approach
(zero-shot → patch / localised).

*Enter a free-text change query, pick a dataset / encoder / scoring approach, and get
ranked before→after pairs with a change heatmap on T2.* To (re)generate the screenshot
locally (it lands in `report/figures/app_screenshot.png`):

```bash
pip install -e .
python -m scripts.make_den_fixture
python -m src.app --root tests/fixtures/den_tiny --split all --encoder clip_vitl14   # http://127.0.0.1:7860
# then save a screenshot of the browser tab to report/figures/app_screenshot.png
```

## Pipeline

End-to-end flow for one user query, with the module responsible at each
step:

```
┌─ offline (one-time per dataset+encoder, cached) ─────────────────────────┐
│                                                                          │
│  TemporalDataset.list_pairs()                src/datasets/*               │
│           │                                                               │
│           ▼                                                               │
│  load_pair_images(pair)  ──▶  PIL T1, T2                                  │
│           │                                                               │
│           ▼                                                               │
│  ImageTextEncoder.encode_image     src/encoders/*  (CLIP / GeoRS /        │
│           │                                          RemoteCLIP, frozen)  │
│           ▼                                                               │
│  f_T1, f_T2  (L2-normed, [N, D])  ──cache──▶                              │
│   data/cache/<dataset>__<encoder>__<split>[_<color>][_lora]__pair_embeddings.npz │
│                                       src/embeddings.py                   │
└──────────────────────────────────────────────────────────────────────────┘

┌─ adapter training (only for `peft` approach; offline) ───────────────────┐
│                                                                          │
│  weak caption per pair  ──ProjectionHead──▶  masked symm. InfoNCE         │
│  (e.g. "agriculture replaced by impervious surface")                     │
│                                              src/train.py                 │
│  → models/<dataset>__<encoder>__adapter.pt                                │
└──────────────────────────────────────────────────────────────────────────┘

┌─ inference (per query, hot path) ────────────────────────────────────────┐
│                                                                          │
│  user query text  ─encoder.encode_text─▶  t  (same shared [D] space)     │
│                                                                          │
│       ┌───────────────── ChangeRetriever.score_all ──────────────────┐    │
│       │   naive      :  t · f_T2                                     │    │
│       │   zero_shot  :  t · f_T2  −  t · f_T1   (Δ-similarity)       │    │
│       │   peft       :  t · g(Δf)   with Δf = f_T2 − f_T1            │    │
│       └────────────────────────────────────────────────────────────┘    │
│                              src/retrieval.py                            │
│                                    │                                     │
│                                    ▼                                     │
│  rank all pairs by score  ──▶  top-K change events                       │
│                                    │                                     │
│                                    ▼                                     │
│  for top-1: dataset.load_pair_images(pair)  +                            │
│  encoder.compute_patch_text_similarity → heatmap on T2                   │
│                                              src/app.py + src/heatmap.py │
│                                                                          │
│  label-grounded benchmark (offline, optional)                            │
│  per-query relevance from PairLabel → Recall@K, mAP, seasonal drift      │
│                                              src/benchmark.py            │
└──────────────────────────────────────────────────────────────────────────┘
```

Three scoring **approaches** (the supervisor-requested comparison):

| Approach    | Score                                       | Training |
|-------------|---------------------------------------------|----------|
| `naive`     | cos(t, f_T2)                                | none (lower bound) |
| `zero_shot` | cos(t, f_T2) − cos(t, f_T1)  (Δ-similarity) | none |
| `peft`      | cos(t, g(Δf)), g = trained ProjectionHead   | ~0.5–0.7 M params (adapter only; backbones frozen) |

*Per-encoder results for all three approaches — in-distribution and cross-split — are in the report ([`report/main.pdf`](report/main.pdf), §8.2).*

**Key decoupling:** `f_T1, f_T2` are cached per `(dataset, encoder, split, color_mode)` so all
three approaches and any number of queries reuse the same one-time encode
pass. Adding a dataset = implementing the `TemporalDataset` protocol +
registering — the entire flow above re-uses the new dataset unchanged
(see [Extending](#extending) below).

**Evaluation** is label-grounded: a fixed query set (per dataset, under
`src/queries/<name>.py`) maps each query to a relevance rule over the
derived `PairLabel`s → Recall@K, mAP, plus a seasonal-vs-permanent
("semantic drift") error report.

## Results at a glance

Frozen vision-language change retrieval hits a **robust ≈0.20 cross-validated-mAP ceiling** on
Dynamic EarthNet — best configuration **GeoRSCLIP + NRG with patch-level top-3 Δ-scoring: CV mAP
0.193 ± 0.051** (4/9 queries FDR-significant), recovery scaling with the visual salience of the
change. Open-vocabulary breadth holds on LEVIR-CC / SECOND-CC (salient building/urban change
strong; subtle/sparse change weak); PEFT/LoRA adapters memorise training AOIs with no held-out
gain over frozen zero-shot; heatmap localisation is a weak signal. All numbers are audited —
random-ranking baselines, permutation tests, BH-FDR, and leakage-free 5-fold leave-AOI-out
cross-validation.

**Full results** — per-encoder tables, the honest single-split→CV arc, every ablation, and all
figures — **are in the deliverable report: [`report/main.pdf`](report/main.pdf).**

## Module map

| File | Role |
|------|------|
| `src/datasets/` | `TemporalDataset` protocol, `DENDataset` (raster), `DENNpyDataset` (DynNet `.npy` + `color_mode` rgb/nrg/ndvi via NIR infrared frames), `TEOChatlasQFabricDataset` (`qfabric_teo` — QFabric crops + RQA2 change-type labels), `StatusQFabricDataset` (`qfabric_status` — RQA5 status transitions), `LevirCCDataset` (`levir_cc` — building-change pairs + human captions), `LevirMCIDataset` (`levir_mci` — LEVIR-CC + building/road change masks), `SecondCCDataset` (`second_cc` — captioned six-class land-cover change + per-phase semantic masks), `DENPlanetDataset` (`dynamic_earthnet_planet` — native 3 m Planet-Fusion surface-reflectance rasters), layout-detecting registry + opts adapters |
| `src/queries/` | Per-dataset query sets (`den.py`, `qfabric.py`, `qfabric_status.py`, `levir_cc.py`, `levir_mci.py`, `second_cc.py`); registry resolved by `dataset.name` |
| `src/results_io.py` | serialize `BenchmarkReport` to JSON/CSV (torch-free); consumed by the figure scripts |
| `src/error_analysis.py` | per-query confusion matrix + precision/recall (seasonal-vs-permanent error analysis) |
| `src/encoders/` | `ImageTextEncoder` protocol; `clip_vitl14` (768-d), `georsclip` (512-d), `remoteclip` (768-d) |
| `src/text_encoder.py` | frozen CLIP text tower (`text_model` + `text_projection`, device-aware) |
| `src/features.py` | `compute_change_feature` (difference / concatenate) |
| `src/embeddings.py` | per-pair `f_T1,f_T2` compute + npz cache (`PairEmbeddingStore`); `cache_tag` arg keys cache by split+color to prevent cross-split collision |
| `src/retrieval.py` | `ChangeRetriever` — naive / zero_shot / peft scoring |
| `src/benchmark.py` | query set + label relevance, Recall@K / mAP / drift |
| `src/model.py` | `ProjectionHead` adapter, InfoNCE, adapter save/load |
| `src/train.py` | PEFT training (masked symmetric InfoNCE on weak captions) |
| `src/lora_train.py` | LoRA fine-tuning of visual encoder via peft; `train_lora`, `merge_lora_into_encoder`, `save_lora` |
| `src/geo_filter.py` | `GeoFilter` — filter pairs by continental region or lat/lon bbox using `aoi_metadata.json`; toggleable |
| `src/rerank.py` | `Reranker` — post-retrieval re-ranking: `diversity` (unique AOIs) or `coherence` (cluster near top-1); toggleable |
| `src/app.py` | Gradio engine + UI (Dataset / Encoder / Approach selectors) |
| `app.py` | HuggingFace Spaces entry point (uses tiny fixture by default; override via env vars) |
| `scripts/download_den.py` | fetch + extract DEN subset, build label index |
| `scripts/build_qfabric_labels.py` | TEOChatlas RQA2 → `qfabric_teo_labels.json` (27,879 real crop→change-type labels) |
| `scripts/build_qfabric_status_labels.py` | TEOChatlas RQA5 → `qfabric_status_labels.json` (per-timepoint construction-status labels) |
| `scripts/eval_rerank.py` | re-ranking benchmark (diversity / coherence) on the DEN test split (report Appendix C) |
| `scripts/make_cv_figure.py` | CV-progression figure (single-split → full-corpus → 5-fold) from `results/` (report §8.1) |
| `scripts/run_seasonal_gate.py` | seasonal false-positive gate / stable-pair FPR robustness check |
| `scripts/benchmark_qfabric.py` | extract QFabric crops + encode + label-grounded change-type mAP (`qfabric_teo`) |
| `scripts/benchmark_levir_cc.py` | LEVIR-CC 5-query open-vocab retrieval, per-query AP (reads the shared LEVIR-MCI dir) |
| `scripts/benchmark_second_cc.py` | SECOND-CC 7-query open-vocab breadth retrieval, per-query AP |
| `scripts/eval_localization.py` | quantitative change localisation (pointing-game + patch-AP vs mask) — `--dataset levir_mci\|second_cc` |
| `scripts/peft_augment_eval.py` | anti-memorization check: frozen / PEFT / PEFT+feature-noise on the same leakage-free folds (exploratory; not surfaced in the report) |
| `scripts/make_den_fixture.py` | tiny synthetic DEN tree for fast tests |
| `scripts/run_pipeline.py` | one-command run with `--train-split` / `--eval-splits` / `--color-mode` / `--mode` / `--lora` / `--results-dir`; cross-split mAP table |
| `scripts/precompute_patch_embeddings.py` | warm the on-disk per-patch embedding cache (`PatchEmbeddingStore` in `src/embeddings.py`) so the first `approach="patch"` query in the app is instant instead of a full GPU pass |
| `scripts/export_results.py` | regenerate benchmarks from cache → `results/*.json` + `macro_summary.csv` (`--confusion` for error analysis) |
| `scripts/make_figures.py` | publication PNGs (recall curves, mAP bars, colour heatmap, seasonal drift, cross-split, confusion) from `results/` |
| `scripts/make_comparison_figure.py` | static zero-shot-vs-PEFT top-K visual comparison per encoder |
| `scripts/make_qualitative_figure.py` | qualitative salient-vs-subtle retrieval figure — actual top-1 pair per query as [Before \| After \| change heatmap], relevance from the query predicate (honest, non-cherry-picked; heatmap shown as a weak localiser, report §8.5) |
| `scripts/lora_sweep.py` | LoRA rank/epoch sweep (georsclip+nrg), in-memory, no cache/model clobber |
| `scripts/significance_audit.py` | random-ranking baseline + permutation p + BH-FDR over every result cell → `results/results_audit_summary.csv` (report §7 protocol, applied across §8) |
| `scripts/cv_eval.py` | full-corpus + k-fold AOI cross-validation with bootstrap CIs; `--relevance fraction` swaps dominant-class-flip relevance for pixel-fraction (report §8.1); merges cached split embeddings, no re-encode |
| `scripts/patch_eval.py` | patch-level (localised) Δ-similarity change retrieval vs the global baseline (report §8.1, "S3"); caches per-patch embeddings via `encoder.encode_image_patches`. `--approach hybrid` fuses global+patch, `patch_softattn`/`patch_spatial` are training-free change-attention variants, `--prompt-ensemble` averages query templates (all report §8.3) |

*Native-3m experiment scripts — `download_den_planet`, `make_jpeg_ablation_figure`, `make_temporal_pinpoint_figure` — are documented in [`feature_3m_native/README.md`](feature_3m_native/README.md); `make_dataset_figures` and `make_pipeline_figure` generate report figures from `results/`; `scripts/deploy_space.py` publishes the HuggingFace Space.*

## Run / install / use

**Requirements:** Python 3.12+ · ~3 GB disk (model weights) · ~9 GB more for real DEN · GPU optional.
RS-encoder weights download from HuggingFace on first use into `.model_cache/`. On-disk DEN
layouts (`planet/*.tif` raster or DynNet preprocessed `.npy`) are auto-detected; dataset sources
in [Datasets & model resources](#datasets--model-resources) below.

### 1. Setup (one-time)

```bash
git clone https://github.com/Panagiotis427/msc-gbda-ov-temporal-change-retrieval.git && cd msc-gbda-ov-temporal-change-retrieval

python -m venv .venv
source .venv/bin/activate          # Windows (PowerShell): .venv\Scripts\Activate.ps1
                                   #   if blocked once: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

pip install -e .
```

### 2. Option A — 30-second synthetic demo (no download)

```bash
python -m scripts.make_den_fixture
# Builds tests/fixtures/den_tiny/: 2 AOIs × 8 months, <1 MB, deterministic.

python -m src.app --root tests/fixtures/den_tiny --split all --encoder clip_vitl14
# First run downloads CLIP weights (~1.6 GB) into .model_cache/ — one-time.
# Open http://127.0.0.1:7860
```

### 2. Option B — real Dynamic EarthNet (~7 GB)

```bash
python -m scripts.download_den --dest data/DynamicEarthNet
# ~7 GB ZIP via gdown; extracted; idempotent (_done.marker guards re-runs).

python -m src.app --dataset dynamic_earthnet --root data/DynamicEarthNet --encoder clip_vitl14
# DEN's launch profile supplies split/colour; --approach defaults to zero_shot.
# Switch to --approach peft (or patch) in the UI for the other scorings.
```

### App usage

Type a free-text change query (or click a curated example) and press **Search**. The top match opens
in a detail panel with two swipe sliders — before against after, and after against the
query-conditioned change heatmap — beside its match score and a seasonal-vs-permanent note, with
buttons to download the before, after, and heatmap images. The match score is the raw retrieval
score (a difference of cosine similarities) min–max normalised across the returned results: a
relative ranking within the query, not a calibrated probability. The remaining matches fill a thumbnail
grid whose per-tile **View** button promotes any result into the detail panel, and a ranked table
(exportable to CSV) lists them all. A collapsible **Settings** panel chooses the dataset, encoder,
colour mode, PEFT/LoRA, and the optional geographic filter and re-ranking, applied on **Apply** —
for that browser session only, so visitors to a shared deployment do not change each other's corpus; a
collapsible **About** panel explains each scoring approach and the honest accuracy expectations.
Startup defaults are set through CLI flags:

| Flag | Default | Notes |
|---|---|---|
| `--dataset` | `levir_mci` | Corpus to load (must have a query set + launch profile). |
| `--encoder` | `georsclip` | `clip_vitl14` / `georsclip` / `remoteclip`. |
| `--approach` | `zero_shot` | `naive` / `zero_shot` / `patch` / `peft` (switchable in the UI). |
| `--root` | dataset profile | Dataset dir; defaults to the selected dataset's launch-profile root. |
| `--split` | dataset profile | `train`/`val`/`test`/`all`; defaults to the dataset's profile split. |
| `--color-mode` | dataset profile | `rgb`/`nrg`/`ndvi`; DEN profile defaults to `nrg`, other corpora to `rgb`. |
| `--pairing` | `bimonthly` | How DEN's 24 monthly timesteps pair into (T1, T2). |
| `--host` | `127.0.0.1` | Bind address; `0.0.0.0` exposes on the LAN (auto-selected on a HF Space). |
| `--port` | `7860` | Gradio HTTP port (in use? add `--port 7861`). |
| `--lora` / `--no-lora` | off | Load LoRA-adapted embeddings (pre-cache via `run_pipeline --lora`). |
| `--geo-filter` / `--no-geo-filter` | off | Geographic region filter. |
| `--rerank` / `--no-rerank` | off | Post-retrieval re-ranking. |
| `--rerank-strategy` | `diversity` | `diversity` = unique locations; `coherence` = cluster near top-1. |

> `peft` errors "no adapter" → adapter missing from `models/`; train with `run_pipeline` (below)
> or switch to `zero_shot`. **Hosted demo (no install):** the app is live on the
> [HuggingFace Space](https://huggingface.co/spaces/panagiotis427/Open_Vocabulary_Temporal_Change_Retrieval);
> redeploy the current tree with `python scripts/deploy_space.py`.

### Developer — pipeline, training, tests

```bash
# Full pipeline: train on train split, evaluate on all three splits
python -m scripts.run_pipeline --root data/DynamicEarthNet \
    --encoder clip_vitl14 --train-split train --eval-splits train val test --epochs 40

# Best zero-shot generalisation (GeoRSCLIP + NIR, no training)
python -m scripts.run_pipeline --root data/DynamicEarthNet \
    --encoder georsclip --color-mode nrg --eval-splits train val test --skip-train

# LoRA adapter on the visual encoder
python -m scripts.run_pipeline --root data/DynamicEarthNet \
    --encoder georsclip --color-mode nrg --skip-train \
    --lora --lora-epochs 20 --lora-rank 4 --lora-alpha 8 --eval-splits train val test
```

Repeat with `--encoder georsclip` / `remoteclip` for the three-encoder comparison in the report
([`report/main.pdf`](report/main.pdf), §8.2). `run_pipeline` is the canonical, cache-consistent flow; the
individual stages (`src.embeddings`, `src.benchmark`, `src.train`, `src.lora_train`) are
convenience entry points — pass the same `--split` / `--color-mode` to every stage so they
share the split-tagged embedding cache.

```bash
pytest -q                              # full suite (test_text_encoder needs the real CLIP weights — network or a warm cache — and skips without them)
pytest -q --ignore=tests/test_text_encoder.py   # fast CPU loop, ~65 s (mock encoders, synthetic fixture)
```

## Dependencies — `pyproject.toml` vs `requirements.txt`

Two dependency files, two purposes:

- **`pyproject.toml`** — the full development install (`pip install -e .`). It is the source of
  truth for local work: the runtime stack **plus** the test/figure/data extras the app itself
  never imports (`pytest`, `coverage`, `matplotlib` for the report figures, `gdown` for dataset
  download) and `opencv-python`. `pip` ignores the `[tool.uv.sources]` CUDA index, so a bare
  editable install pulls CPU Torch — install the matching CUDA wheel afterwards if you want GPU.
- **`requirements.txt`** — the lightweight **HuggingFace Space** deployment subset: just the
  runtime the `app.py` import path needs. It drops the test/figure/data extras and uses
  `opencv-python-headless` (no display libs) to keep the Space image small.

Keep the runtime packages consistent between the two; only the test/figure/data extras and the
`opencv-python` → `opencv-python-headless` swap should differ.

## Extending

The pipeline is **dataset-agnostic**. Every shared file (`embeddings.py`, `retrieval.py`,
`benchmark.py`, `train.py`, `app.py`, `scripts/run_pipeline.py`) consumes only the
`TemporalDataset` protocol and the dataset/encoder/query registries. Concrete loaders and
dataset-specific choices live in their own modules and self-register.

**Hard rule: adding a dataset = adding files only, never editing shared pipeline files.**

### Plug-in points

| Concern | Where | How |
|---|---|---|
| Dataset loader | `src/datasets/<name>.py` | Implement the `TemporalDataset` protocol (see `src/datasets/base.py`) |
| Loader registration | same file | `register_dataset(name, factory, opts_adapter)` from `src/datasets/registry.py`; add a line to `registry.py` only for a new built-in — third-party datasets can register from their own module on import |
| Generic-options mapping | same opts adapter | Maps `(root, pairing, split, **extra)` → loader kwargs; `color_mode` travels via `**extra` to `DENNpyDataset` |
| Encoder | `src/encoders/<name>.py` | Implement `ImageTextEncoder`; `register_encoder(...)` in `src/encoders/__init__.py` |
| Benchmark query set | `src/queries/<name>.py` | List of `Query(text, category, predicate)`; `register_queries(name, queries)`; auto-imported by `src/queries/__init__.py` |
| CLI (`--dataset` / `--encoder`) | nothing | Choices derive from the dataset/encoder registries — a registry-only dataset is immediately usable from `scripts/run_pipeline.py`, `scripts/export_results.py`, and `python -m src.app --dataset <name>` |
| App dropdown | `src/app.py` (`DATASET_PROFILES`) | The **one** shared-file touch a dataset needs: the Gradio demo needs a launch profile (label, root, split, colour, rank) per dataset, so a registry-only dataset is CLI-usable but does **not** appear in the dropdown until it has a `DATASET_PROFILES` entry. Deliberate exception to the files-only rule, scoped to the demo UI |

### What a new dataset adds (and only adds)

```
src/datasets/<name>.py          # loader, register_dataset(...)
src/queries/<name>.py           # query set, register_queries(...)
src/queries/__init__.py         # ONE import line: from . import <name>
tests/test_<name>.py            # loader-level tests
```

If you find yourself editing `embeddings.py`, `retrieval.py`, `benchmark.py`, `train.py`,
`app.py`, or `scripts/run_pipeline.py` for a new dataset, stop — an existing extension point
already covers it.

### Cache & artefact paths

- **Embeddings:** `data/cache/<dataset>__<encoder>[__<tag>]__pair_embeddings.npz`, where `<tag>` =
  `{split}[_{color_mode}][_lora]` — built by `cache_tag_for()`. Pass `cache_tag` to
  `load_or_compute()` to isolate caches per split / colour / LoRA.
- **Adapters:** `models/<dataset>__<encoder>[_<color>][_<split>][_<mode>]__adapter.pt` — the
  committed `train` split + `difference` mode take no suffix; others append `_<color>` / `_<split>` /
  `_<mode>` (single underscore each). `scripts/export_results.py --train-split/--mode` locate the
  matching adapter for a non-default run.
- Keyed by `(dataset, encoder, split, color_mode)` — no cross-split/colour collision; a stale
  pair-set on load triggers automatic recompute and overwrites the cache at the same path.

### Shared helpers (not plug-in points)

- `src/stats.py::rand_ap(...)` — the shuffle-based random-AP baseline used by the significance
  scripts. `scripts/cv_eval.py` keeps its own `rng.permutation` variant on purpose, to preserve its
  committed RNG-dependent results.
- `src/embeddings.py::cache_tag_for(split, color_mode, lora)` and its sibling `color_tag(color_mode)`
  — the single source of truth for split/colour/LoRA cache tags and the `_<color>` artefact suffix;
  import them rather than re-deriving the strings (every pipeline/benchmark/export script does).

## Datasets & model resources

Download links and citations for the datasets and encoders used.

### Datasets

- **Dynamic EarthNet** (primary; CVPR 2022, Toker et al.) — daily multi-spectral Planet imagery,
  75 AOIs, monthly 7-class LULC labels. [Paper](https://arxiv.org/abs/2203.12560) ·
  [dynnet repo](https://github.com/aysim/dynnet) · preprocessed ~7 GB via
  `gdown 1cMP57SPQWYKMy8X60iK217C28RFBkd2z` (wrapped by `scripts/download_den.py`) ·
  [torchgeo HF](https://huggingface.co/datasets/torchgeo/dynamic_earthnet) ·
  [HEVC-compressed HF](https://huggingface.co/datasets/tacofoundation/DynamicEarthNet-video) ·
  raw ~525 GB [TUM Mediatum](https://mediatum.ub.tum.de/1650201). Licence: CC BY-SA 4.0 (per mediaTUM).
- **QFabric** (CVPR EarthVision 2021, Verma et al.) — used here in the reduced 2-date TEOChatlas
  form (`qfabric_teo`): [TEOChatlas](https://huggingface.co/datasets/jirvin16/TEOChatlas). The full
  5-date + COCO-polygon-mask form ([labaerien/qfabric](https://huggingface.co/datasets/labaerien/qfabric),
  **gated**) is access-blocked and out of scope — see the report's §11.
  [Paper](https://openaccess.thecvf.com/content/CVPR2021W/EarthVision/papers/Verma_QFabric_Multi-Task_Change_Detection_Dataset_CVPRW_2021_paper.pdf).
  Licence: TEOChatlas is published as Apache-2.0; the full QFabric as CC BY-NC 4.0 (per their HF cards).
  (Avoid `EVER-Z/QFabric_mt_images_1024` — 298 GB, image-only, no masks.)
- **LEVIR-CC / LEVIR-MCI** — building/road change captions + pixel masks (in-repo loaders
  `levir_cc` / `levir_mci`). Built on LEVIR-CD's Google Earth imagery: academic use only, no
  commercial use ([LEVIR-CD terms](https://justchenhao.github.io/LEVIR/)), and Google Earth's terms apply.
- **SECOND-CC** — six-class land-cover change + semantic maps
  ([Zenodo 10.5281/zenodo.16937571](https://doi.org/10.5281/zenodo.16937571); `second_cc`). Licence: CC BY 4.0.
- **fMoW** (CVPR 2018) — assessed and **rejected** (functional classification, no change labels;
  see the report's §11). [Paper](https://arxiv.org/abs/1711.07846).

### Encoders

- **CLIP ViT-L/14** (OpenAI, Radford et al. 2021) — general backbone.
  [Paper](https://arxiv.org/abs/2103.00020) · [HF](https://huggingface.co/openai/clip-vit-large-patch14).
  Licence: MIT ([openai/CLIP](https://github.com/openai/CLIP)).
- **GeoRSCLIP** (RS5M, Om AI Lab) — RS-pretrained, the headline encoder.
  [Paper](https://arxiv.org/abs/2306.11300) · [HF](https://huggingface.co/Zilun/GeoRSCLIP).
  Licence: its HF card says only "cc"; the RS5M data it was trained on is CC BY-NC 4.0.
- **RemoteCLIP** (IEEE TGRS) — RS-pretrained.
  [Paper](https://arxiv.org/abs/2306.11029) · [repo](https://github.com/ChenDelong1999/RemoteCLIP).
  Licence: Apache-2.0 (the repo; its HF weights card states none).
- All loaded via [OpenCLIP](https://github.com/mlfoundations/open_clip); weights auto-download from
  HuggingFace on first use into `.model_cache/`.

---

## Report

The complete technical account — methodology, the full statistical protocol, every ablation, the native-3m data-source-fidelity check, temporal pinpointing, and per-dataset results — is the **compiled deliverable report, [`report/main.pdf`](report/main.pdf)**, tracked in the repo so it reads directly on GitHub with no LaTeX build (LaTeX source: [`report/main.tex`](report/main.tex)). Its headline is a robust ≈0.20 cross-validated-mAP ceiling for frozen vision-language change retrieval (best configuration GeoRSCLIP + NRG with patch-level scoring, 0.193 ± 0.051), with recovery scaling by the visual salience of the change; the [Results at a glance](#results-at-a-glance) summary above is the audited in-repo digest.

*Authors, the full acknowledgements, and the complete reference list are on the report's title page and bibliography.*

## License

The code in this repository is released under the [MIT License](LICENSE). That licence does not extend to third-party data or weights: the datasets and encoders keep the terms listed under [Datasets & model resources](#datasets--model-resources), and the adapters in `models/` were trained on Dynamic EarthNet and QFabric/TEOChatlas features from those encoders, so check the upstream terms (several are share-alike or non-commercial) before reusing them outside research. The technical report in `report/` and its figures are the authors' academic work — please cite the report rather than redistribute it.
