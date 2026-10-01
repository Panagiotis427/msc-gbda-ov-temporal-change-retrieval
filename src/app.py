"""
Semantic Change Search Engine — Gradio frontend (rewired for real change
retrieval).

A natural-language query is scored against every bi-temporal pair's *change*
representation (not single images), returning a ranked list of change events.
Each result shows the actual T1/T2 tiles of the matched pair, a
query-conditioned change heatmap (per-patch Δ-similarity T1→T2) on the After
tile, a relative match score, and a seasonal-vs-permanent note grounded in the dataset's
labels.

Selectors: Dataset, Encoder (CLIP / GeoRSCLIP / RemoteCLIP), Approach
(naive / zero_shot Δ-similarity / patch (localised per-patch Δ) / peft adapter).

CLI:
    python -m src.app --dataset dynamic_earthnet --root data/DynamicEarthNet \
        --encoder clip_vitl14 --approach zero_shot
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import tempfile
import threading
import traceback
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image

from src.embeddings import load_or_compute
from src.encoders import get_encoder
from src.retrieval import APPROACHES, ChangeRetriever
from src.heatmap import generate_change_heatmap, generate_heatmap
from src.rerank import RERANK_STRATEGIES

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SNOW = "snow_and_ice"

_CSV_HEADER = ["rank", "location", "before", "after", "score",
               "match_0_1", "caption", "land_cover_change"]


_EXPORT_DIR = None  # one lazily-created temp dir holding every app export
_REQUEST_DIRS_KEPT = 32  # recent searches whose exported files stay on disk
_MAX_RESULTS = 10  # Top-K slider maximum == size of the all-matches grid


def clamp_top_k(value) -> int:
    """Top-K as the app serves it. The slider bounds it only in the browser; the
    API endpoint accepts any number, and every result costs an image load plus a
    heatmap pass, so a huge value would sweep the whole corpus on one request."""
    return max(1, min(int(value), _MAX_RESULTS))


def _safe_name(raw: str) -> str:
    return "".join(c if (c.isalnum() or c in "._-") else "_" for c in str(raw)) or "image"


def _export_dir() -> str:
    """The app's temp root for exports (named images + result CSV)."""
    global _EXPORT_DIR
    if _EXPORT_DIR is None:
        _EXPORT_DIR = tempfile.mkdtemp(prefix="change_app_")
    return _EXPORT_DIR


def _request_dir() -> str:
    """A fresh export subdirectory for one search or View click. Files keep their
    meaningful basenames (Gradio uses them as download names), yet two requests can
    never overwrite each other's heatmap or CSV — the heatmap depends on the query,
    the basename does not. Only the most recent few are kept, so disk use stays
    bounded on a long-running Space."""
    root = _export_dir()
    path = Path(tempfile.mkdtemp(prefix="req_", dir=root))
    # Oldest first; the new directory is never a candidate (same-second mtimes tie).
    others = sorted((p for p in Path(root).iterdir()
                     if p.is_dir() and p.name.startswith("req_") and p != path),
                    key=lambda p: p.stat().st_mtime)
    for old in others[:max(0, len(others) - (_REQUEST_DIRS_KEPT - 1))]:
        shutil.rmtree(old, ignore_errors=True)
    return str(path)


def materialize_image(img, raw_name: str, out_dir: Optional[str] = None) -> Optional[str]:
    """Save a PIL image to a PNG with a meaningful basename and return its path.

    Gradio names a downloaded image after the served file's basename, so passing a
    named path (rather than a bare PIL object, which downloads as ``image.png``)
    gives the user a sensible filename like ``after_2065_2018-07-01.png``.
    ``out_dir`` defaults to the shared export root; the UI passes a per-request
    directory. Returns None when *img* is None.
    """
    if img is None:
        return None
    path = os.path.join(out_dir or _export_dir(), _safe_name(raw_name) + ".png")
    img.save(path)
    return path


def results_to_csv(rows, dataset: str, out_dir: Optional[str] = None) -> Optional[str]:
    """Write ranked result *rows* to a CSV with a clean, dataset-named basename
    (e.g. ``change_results_levir_mci.csv``) so the in-app download has a usable
    name + extension, not a random temp name. Returns the path, or None if no rows.
    """
    if not rows:
        return None
    safe = "".join(c if (c.isalnum() or c in "-_") else "_" for c in dataset) or "results"
    path = os.path.join(out_dir or _export_dir(), f"change_results_{safe}.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(_CSV_HEADER)
        writer.writerows(rows)
    return path


def displayable_events(evs):
    """Split ranked events for the results view.

    Returns ``(shown, top)`` where ``shown`` is the events that have a displayable
    image (``heatmap`` or ``t2_img`` — the all-matches tiles + View buttons index
    these 1:1) and ``top`` is the Top-match event: the first displayable one,
    falling back to ``evs[0]`` only if none are displayable (``None`` if empty).
    This keeps the Top-match panel and the first View button pointing at the same
    event even when a higher-ranked pair's images fail to load.
    """
    shown = [e for e in evs if (e.heatmap or e.t2_img) is not None]
    top = shown[0] if shown else (evs[0] if evs else None)
    return shown, top


def _app_datasets() -> list:
    """Datasets offered in the app UI: those with a registered query set, so
    label-grounded retrieval, relevance and the seasonal note all work. Any
    loader without a query set (no labels / no benchmarked data) is excluded,
    since it would otherwise error if picked.
    """
    from src.queries import get_queries  # importing the package registers the sets
    from src.datasets.registry import list_datasets
    return [d for d in list_datasets() if get_queries(d)]


# Friendly display labels for the dataset / encoder dropdowns. The value sent to
# the callbacks stays the registry key; only the shown label changes. Unknown
# keys fall back to the raw key so a newly-registered dataset/encoder still appears.
DATASET_LABELS = {
    "dynamic_earthnet": "Dynamic EarthNet — land-cover transitions",
    "levir_cc": "LEVIR-CC — building/road (captions)",
    "levir_mci": "LEVIR-CC — building/road change",
    "second_cc": "SECOND-CC — 6-class land cover",
    "qfabric_teo": "QFabric — change type",
    "qfabric_status": "QFabric — construction status (weak)",
}
ENCODER_LABELS = {
    "clip_vitl14": "CLIP ViT-L/14",
    "georsclip": "GeoRSCLIP",
    "remoteclip": "RemoteCLIP",
}


def _labeled(choices, labels):
    """``[(display_label, value)]`` for a Dropdown; value stays the registry key."""
    return [(labels.get(c, c), c) for c in choices]


# Per-dataset launch profile so the in-app Dataset dropdown actually *works* when
# switched (each corpus lives in its own directory with its own split/colour).
# Without this, switching datasets reused the DEN root and failed. ``color_mode``
# is omitted for DEN so its dropdown still selects rgb/nrg/ndvi; the others are
# RGB-only (their loaders ignore colour) and pin the cache tag to the committed
# ``__test`` caches. Datasets *not* listed here are excluded from the app dropdown
# (e.g. QFabric needs extra loader args + a custom-tagged cache — use its
# benchmark scripts). Paths are repo-relative.
_QFABRIC_CROPS = str(_PROJECT_ROOT / "data" / "QFabric" / "teochat_crops")
DATASET_PROFILES = {
    "dynamic_earthnet": {"root": str(_PROJECT_ROOT / "data" / "DynamicEarthNet"),
                         "split": "test", "pairing": "bimonthly"},
    # LEVIR: keep only the MCI superset (identical retrieval to LEVIR-CC in-app,
    # plus masks for the offline localization script). levir_cc stays registered
    # for scripts but is not a separate app dropdown entry.
    "levir_mci": {"root": str(_PROJECT_ROOT / "data" / "_levir_mci" / "extracted" / "LEVIR-MCI-dataset"),
                  "split": "test", "color_mode": "rgb"},
    "second_cc": {"root": str(_PROJECT_ROOT / "data" / "_second_cc" / "extracted" / "SECOND-CC-AUG"),
                  "split": "test", "color_mode": "rgb"},
    "qfabric_teo": {"root": _QFABRIC_CROPS, "split": None, "color_mode": "rgb",
                    "loader_extra": {"labels_path": str(_PROJECT_ROOT / "data" / "QFabric"
                                                        / "qfabric_teo_labels.json"),
                                     "max_per_class": 120}},
    "qfabric_status": {"root": _QFABRIC_CROPS, "split": None, "color_mode": "rgb",
                       "loader_extra": {"labels_path": str(_PROJECT_ROOT / "data" / "QFabric"
                                                           / "qfabric_status_labels.json"),
                                        "max_per_class": 120}},
}

# Peak retrieval mAP achieved per dataset (report §8), used only to sort the app
# dropdown best-first so the strongest corpora surface at the top. (LEVIR salient
# construction ~0.8; SECOND-CC buildings ~0.7; QFabric change-type ~0.27 naive, 0.33 with the PEFT head on
# test; DEN patch_top3 ~0.19; QFabric status ~0.08. Rough sort keys, not results to quote.)
DATASET_RANK = {
    "levir_mci": 0.83, "second_cc": 0.70,
    "qfabric_teo": 0.27, "dynamic_earthnet": 0.19, "qfabric_status": 0.08,
}


def _app_dataset_choices() -> list:
    """Datasets offered in the app dropdown: those with a query set AND a launch
    profile (so switching to them in the UI actually loads them), sorted by best
    achieved result (descending) so the strongest corpora appear first."""
    profiled = [d for d in _app_datasets() if d in DATASET_PROFILES]
    return sorted(profiled, key=lambda d: -DATASET_RANK.get(d, 0.0))


@dataclass
class RunConfig:
    # Default corpus = LEVIR-CC (via the MCI superset loader): the strongest demo
    # (salient building/road change, per-query AP up to ~0.8) for the best first
    # impression. zero_shot uses the cached global embeddings (snappy first query);
    # patch would re-encode 1929 pairs. Other corpora (incl. Dynamic EarthNet, the
    # report's primary analysed dataset) are selectable + sorted best-first in the UI.
    dataset: str = "levir_mci"
    encoder: str = "georsclip"
    approach: str = "zero_shot"
    root: str = str(_PROJECT_ROOT / "data" / "_levir_mci" / "extracted" / "LEVIR-MCI-dataset")
    pairing: str = "bimonthly"
    split: Optional[str] = "test"
    cache_dir: str = str(_PROJECT_ROOT / "data" / "cache")
    feature_mode: str = "difference"
    top_k: int = 5
    # Spectral / encoder options (require Apply to reload embeddings)
    color_mode: str = "rgb"         # rgb | nrg | ndvi (nrg/ndvi only apply to DEN)
    use_lora: bool = False          # load LoRA-adapted embeddings (must be pre-cached)
    # Extension toggles (take effect on next Search; no Apply needed)
    geo_filter: bool = False        # enable geographic region filtering
    rerank: bool = False            # enable post-retrieval re-ranking
    rerank_strategy: str = "diversity"  # diversity | coherence
    # Extra loader kwargs forwarded to build_dataset (e.g. QFabric's labels_path /
    # max_per_class). Empty for the simple root+split datasets.
    loader_extra: dict = field(default_factory=dict)


@dataclass
class ChangeEvent:
    rank: int
    pair_id: str
    location: str
    t1_key: str
    t2_key: str
    score: float
    confidence: float
    caption: str
    seasonal_note: str
    t1_img: Optional[Image.Image]
    t2_img: Optional[Image.Image]
    heatmap: Optional[Image.Image]


def settings_config(base: RunConfig, dataset: str, encoder: str, approach: str,
                    color_mode: str = "rgb", use_lora: bool = False) -> RunConfig:
    """The RunConfig that "Apply settings" asks for, resolved through the dataset's
    launch profile so switching datasets loads the right directory/split/colour.
    DEN honours the colour dropdown; other corpora pin colour (rgb) via the profile."""
    prof = DATASET_PROFILES.get(dataset, {})
    prof_root = prof.get("root")
    # Use the profile's data dir when it exists; otherwise keep the current root
    # (e.g. on the fixture-only HF Space, where the real corpora are absent —
    # switching to a non-fixture dataset then errors gracefully).
    root = prof_root if prof_root and Path(prof_root).exists() else base.root
    return RunConfig(
        dataset=dataset, encoder=encoder, approach=approach,
        root=root, pairing=prof.get("pairing", base.pairing),
        split=prof.get("split", base.split), cache_dir=base.cache_dir,
        color_mode=prof.get("color_mode", color_mode), use_lora=use_lora,
        geo_filter=base.geo_filter, rerank=base.rerank,
        rerank_strategy=base.rerank_strategy,
        loader_extra=prof.get("loader_extra", {}),
    )


def _engine_key(cfg: RunConfig) -> tuple:
    """The RunConfig fields that determine what an engine loads. Approach, top-K,
    filters and re-ranking are per-query choices and do not need a separate engine."""
    return (cfg.dataset, cfg.encoder, cfg.root, cfg.pairing, cfg.split, cfg.cache_dir,
            cfg.feature_mode, cfg.color_mode, cfg.use_lora,
            tuple(sorted((k, repr(v)) for k, v in cfg.loader_extra.items())))


def _loaded_status(cfg: RunConfig, n_pairs: int) -> str:
    lora_note = " + LoRA" if cfg.use_lora else ""
    return (f"Loaded {cfg.dataset} + {cfg.encoder}{lora_note} | "
            f"color={cfg.color_mode} | approach={cfg.approach} | {n_pairs} pairs")


class SemanticChangeSearch:
    _CSS = """
        .top-card {border: 1px solid var(--border-color-primary, #d0d0d0);
                   border-radius: 12px; padding: 14px 18px;
                   background: var(--background-fill-secondary, #fafafa);}
        /* Single-hue intensity ramp, NOT a traffic light: the pill shows a
           relative match rank within the candidate set, not absolute quality,
           so deep->pale blue = stronger->weaker *position* (no green=good /
           red=bad signal that would oversell a weak retrieval). Hue matches
           the #1565c0 stats accent. */
        .conf-pill {display:inline-block; padding:4px 12px; border-radius:999px;
                    background:#1565c0; color:white; font-weight:600;}
        .conf-pill.mid {background:#6fa8dc; color:#10243a;}
        .conf-pill.low {background:#cfe0f2; color:#10243a;}
        #search-btn { min-height: 3.5rem !important; font-size: 1.1rem !important;
                      letter-spacing: 0.03em; }
        .stats-card {border:1px solid var(--border-color-primary,#d0d0d0);
                     border-left:4px solid #1565c0;
                     border-radius:10px; padding:10px 14px;
                     background:var(--background-fill-secondary,#f5f9ff);
                     font-size:0.95em; line-height:1.7;}
        .stats-card code {background:rgba(21,101,192,0.08); padding:1px 6px;
                          border-radius:4px; font-weight:600;}
        .stats-err {border-left-color:#c62828; background:#fff4f4;}
        /* Keep each comparison slider sized to its (square) tile so the swipe
           divider stays bounded to the image instead of travelling into empty
           letterbox margins. */
        .bounded-slider {max-width: 360px !important; margin-left:auto; margin-right:auto;}
        /* Per-image download buttons: bigger, bold, each its own colour-coded box
           so they read as distinct, obvious download actions (not plain links). */
        .dl-row {gap: 12px;}
        /* elem_classes land on the button element itself, so target both the class
           and a nested button to be robust across Gradio versions. */
        .dl-btn, .dl-btn button {font-size: 1.02rem !important; font-weight: 700 !important;
                        border-radius: 10px !important; padding: 12px 10px !important;
                        border-width: 2px !important; border-style: solid !important;}
        .dl-before, .dl-before button {background:#1565c0 !important; border-color:#0d47a1 !important; color:#fff !important;}
        .dl-after,  .dl-after button  {background:#0f7a5a !important; border-color:#0b5c44 !important; color:#fff !important;}
        .dl-heat,   .dl-heat button   {background:#a8560b !important; border-color:#7f4108 !important; color:#fff !important;}
        /* All-matches grid: each "View" button spans its tile's width. */
        .view-btn, .view-btn button {width:100% !important; font-weight:600 !important;}
        /* Tiles are display-only — the View button below is the click target, so
           don't imply the image itself is clickable. */
        .match-tile, .match-tile * {cursor: default !important;}
        """

    def __init__(self, cfg: RunConfig):
        self.cfg = cfg
        self._build(cfg)

    # -- construction ---------------------------------------------------
    @classmethod
    def from_components(cls, dataset, encoder, store, cfg: RunConfig,
                        adapter=None) -> "SemanticChangeSearch":
        self = cls.__new__(cls)
        self.cfg = cfg
        self.dataset = dataset
        self.encoder = encoder
        self.store = store
        self.retriever = ChangeRetriever(store, encoder,
                                         feature_mode=cfg.feature_mode)
        self._adapter_feature_mode = cfg.feature_mode
        if adapter is not None:
            self.retriever.set_adapter(adapter, cfg.feature_mode)
        self._adapter = adapter
        self._patch_t1 = self._patch_t2 = None  # lazy patch-embedding cache
        self._init_extensions(cfg)
        return self

    def _build(self, cfg: RunConfig) -> None:
        from src.datasets.registry import build_dataset
        from src.embeddings import cache_path as _cache_path
        from src.embeddings import cache_tag_for
        self.dataset = build_dataset(cfg.dataset, root=cfg.root,
                                     pairing=cfg.pairing, split=cfg.split,
                                     color_mode=cfg.color_mode,
                                     **cfg.loader_extra)
        self.encoder = get_encoder(cfg.encoder)

        # Canonical cache tag (single source of truth in src.embeddings) so the
        # pre-computed run_pipeline caches are reused. split=None ("all") -> "all".
        split_str = cfg.split or "all"
        cache_tag = cache_tag_for(split_str, cfg.color_mode, cfg.use_lora)

        if cfg.use_lora:
            lora_cache = _cache_path(cfg.cache_dir, self.dataset.name,
                                     self.encoder.name, tag=cache_tag)
            if not lora_cache.exists():
                raise RuntimeError(
                    f"LoRA embedding cache not found: {lora_cache.name}\n"
                    "Train LoRA first:  python -m scripts.run_pipeline --lora ...\n"
                    "Then re-open the app with 'Use LoRA embeddings' enabled."
                )

        self.store = load_or_compute(self.dataset, self.encoder,
                                     cache_dir=cfg.cache_dir,
                                     cache_tag=cache_tag)
        self.retriever = ChangeRetriever(self.store, self.encoder,
                                         feature_mode=cfg.feature_mode)
        self._adapter = self._maybe_load_adapter(cfg)
        if self._adapter is not None:
            self.retriever.set_adapter(self._adapter, self._adapter_feature_mode)
        self._patch_t1 = self._patch_t2 = None  # invalidate lazy patch cache on (re)build
        self._init_extensions(cfg)

    def _init_extensions(self, cfg: RunConfig) -> None:
        """Load geo-filter and reranker if aoi_metadata.json is present."""
        meta_path = Path(cfg.root) / "aoi_metadata.json"
        if meta_path.exists():
            from src.geo_filter import GeoFilter
            from src.rerank import Reranker
            self._geo_filter = GeoFilter(meta_path)
            self._reranker = Reranker(meta_path)
        else:
            self._geo_filter = None
            self._reranker = None

    def _maybe_load_adapter(self, cfg: RunConfig):
        # Adapter filename must match run_pipeline's save convention, which is
        # colour-tagged: models/<dataset>__<encoder>[_<color>]__adapter.pt.
        # Without the colour tag the app would silently apply the RGB-trained
        # adapter to NRG/NDVI embeddings (a dim-compatible but wrong head).
        from src.model import adapter_path
        fname = adapter_path(cfg.dataset, cfg.encoder, cfg.color_mode).name
        path = Path(cfg.cache_dir).parent / "models" / fname
        if not path.exists():
            path = _PROJECT_ROOT / "models" / fname
        if path.exists():
            try:
                from src.model import load_adapter
                adapter, meta = load_adapter(str(path))
                # Use the feature mode the adapter was TRAINED with, not cfg's —
                # a concatenate-trained head fed a difference (D-dim) vector, or
                # vice-versa, would otherwise mismatch the head's input dim.
                self._adapter_feature_mode = meta.get("feature_mode", cfg.feature_mode)
                print(f"Loaded PEFT adapter: {path} (feature_mode={self._adapter_feature_mode})")
                return adapter
            except Exception as exc:
                print(f"Adapter load failed ({exc}); PEFT disabled.")
        self._adapter_feature_mode = cfg.feature_mode
        return None

    # -- patch (localised) scoring --------------------------------------
    def _patch_scores(self, text: str) -> np.ndarray:
        """Localised patch-level Δ-similarity (report §8.1, best DEN
        config). Per-patch embeddings for the whole corpus are loaded from the
        on-disk patch cache when warm (instant first query) and only computed +
        cached on a miss; the result is held on the engine for later queries.
        The patch rows align with ``self.store.pairs`` (same pair list as the
        pair store), so ``query()`` can index ``_patch_t1[i]`` by store index."""
        from src.benchmark import encode_query
        from src.embeddings import cache_tag_for, load_or_compute_patches
        from src.retrieval import top_patch_change_scores
        if getattr(self, "_patch_t1", None) is None:
            split_str = self.cfg.split or "all"
            # NB: no lora component in the tag — the patch path encodes with the plain
            # self.encoder (LoRA is never merged into it here), so a "_lora"-tagged patch cache
            # would mislabel plain-encoder embeddings as adapted. Patch is encoder-LoRA-agnostic.
            cache_tag = cache_tag_for(split_str, self.cfg.color_mode)
            patch_store = load_or_compute_patches(
                self.dataset, self.encoder, self.store.pairs,
                cache_dir=self.cfg.cache_dir, cache_tag=cache_tag,
                batch_size=32, progress=True,
            )
            self._patch_t1 = patch_store.patch_t1
            self._patch_t2 = patch_store.patch_t2
        t = encode_query(self.encoder, text)
        return top_patch_change_scores(self._patch_t1, self._patch_t2, t)

    # -- query ----------------------------------------------------------
    def query(
        self,
        text: str,
        approach: str,
        top_k: int,
        *,
        geo_region: str = "All",
        rerank_strategy: Optional[str] = None,
    ) -> List[ChangeEvent]:
        """Retrieve top-K change events matching *text*.

        Parameters
        ----------
        text: Natural-language change query.
        approach: ``"naive"``, ``"zero_shot"``, or ``"peft"``.
        top_k: Number of results to return.
        geo_region: If not ``"All"``, restrict pairs to that region
            (requires ``aoi_metadata.json`` to have been loaded).
        rerank_strategy: ``"diversity"``, ``"coherence"``, or ``None``
            (disabled).  Requires ``aoi_metadata.json``.
        """
        if approach == "patch":
            scores = self._patch_scores(text)
        else:
            if approach == "peft" and self.retriever.adapter is None:
                raise RuntimeError(
                    "PEFT selected but no adapter found. Train one with "
                    "`python -m src.train` or pick zero_shot.")
            scores = self.retriever.score_all(text, approach=approach)

        # Geographic filter — mask excluded pairs with -inf
        if geo_region not in ("All", "", None) and self._geo_filter is not None:
            allowed = {
                p.location_id
                for p in self._geo_filter.filter_by_region(self.store.pairs, geo_region)
            }
            mask = np.array([p.location_id in allowed for p in self.store.pairs])
            scores = np.where(mask, scores, -np.inf)

        # Confidence bounds over finite scores only
        finite = scores[np.isfinite(scores)]
        lo = float(finite.min()) if len(finite) else 0.0
        hi = float(finite.max()) if len(finite) else 1.0

        # Re-ranking or default (stable) argsort
        if rerank_strategy is not None and self._reranker is not None:
            order = self._reranker.rerank(
                scores, self.store.pairs, top_k, strategy=rerank_strategy
            )
        else:
            order = np.argsort(-scores, kind="stable")[:top_k]
            order = order[np.isfinite(scores[order])]      # a filtered-out pair is never a result

        # For the patch approach the whole corpus's per-patch embeddings are already
        # cached (in _patch_scores), so reuse them for the change-heatmap instead of
        # re-encoding each shown pair's patches a second time.
        patches_cached = (approach == "patch"
                          and getattr(self, "_patch_t1", None) is not None)

        events: List[ChangeEvent] = []
        for rank, i in enumerate(order):
            pair = self.store.pairs[i]
            t1 = t2 = hm = None
            try:
                t1, t2 = self.dataset.load_pair_images(pair)
            except Exception as exc:
                print(f"image load failed for {pair}: {exc}")
            if t1 is not None and t2 is not None:
                try:
                    # Real query-conditioned CHANGE heatmap (per-patch Δ T1->T2);
                    # fall back to query-vs-After match if no patch tokens.
                    pp1 = self._patch_t1[i] if patches_cached else None
                    pp2 = self._patch_t2[i] if patches_cached else None
                    _, hm = generate_change_heatmap(np.array(t1), np.array(t2),
                                                    text, self.encoder, alpha=0.5,
                                                    precomputed_p1=pp1, precomputed_p2=pp2)
                    if hm is None:
                        _, hm = generate_heatmap(np.array(t1), np.array(t2),
                                                 text, self.encoder, alpha=0.5)
                except Exception as exc:
                    print(f"heatmap failed: {exc}")
            conf = (float(scores[i]) - lo) / (hi - lo) if hi > lo else 1.0
            caption, note = self._describe(pair)
            events.append(ChangeEvent(
                rank=rank + 1,
                pair_id=f"{pair.location_id}_{pair.t1_key}_{pair.t2_key}",
                location=pair.location_id, t1_key=pair.t1_key,
                t2_key=pair.t2_key, score=float(scores[i]),
                confidence=round(conf, 3), caption=caption,
                seasonal_note=note, t1_img=t1, t2_img=t2, heatmap=hm,
            ))
        return events

    def _describe(self, pair) -> tuple[str, str]:
        try:
            label = self.dataset.get_pair_label(pair)
        except Exception:
            label = None
        cap_fn = getattr(self.dataset, "text_caption_for_pair", None)
        caption = cap_fn(pair) if cap_fn else (
            label.change_type if label else "unknown change")
        if label is None:
            return caption, "no label (unlabeled dataset)"
        if _SNOW in (label.dominant_t1_class, label.dominant_t2_class):
            return caption, "involves snow/ice - likely SEASONAL, not permanent"
        if label.stable:
            return caption, "labelled stable - weak/no change"
        return caption, "permanent land-cover change"

    # -- gradio ---------------------------------------------------------
    def stats_markdown(self) -> str:
        """One-line corpus stats card (rendered as Gradio markdown)."""
        n_pairs = len(self.store)
        n_locs = len(self.dataset.list_locations())
        per_loc = (n_pairs / n_locs) if n_locs else 0
        adapter = "loaded" if self.retriever.adapter is not None else "none"
        geo_status = "ready" if self._geo_filter is not None else "N/A (no metadata)"
        rerank_status = "ready" if self._reranker is not None else "N/A (no metadata)"
        return (
            f"<div class='stats-card'>"
            f"<b>Dataset:</b> <code>{self.cfg.dataset}</code> &nbsp;|&nbsp; "
            f"<b>Split:</b> <code>{self.cfg.split or 'all'}</code> &nbsp;|&nbsp; "
            f"<b>Pairing:</b> <code>{self.cfg.pairing}</code> &nbsp;|&nbsp; "
            f"<b>Color:</b> <code>{getattr(self.dataset, 'color_mode', None) or 'rgb'}</code> &nbsp;|&nbsp; "
            f"<b>Locations:</b> <code>{n_locs}</code> &nbsp;|&nbsp; "
            f"<b>Image pairs:</b> <code>{n_pairs}</code> "
            f"({per_loc:.0f}/location) &nbsp;|&nbsp; "
            f"<b>Encoder:</b> <code>{self.encoder.name}</code> "
            f"({self.encoder.embed_dim}-d) &nbsp;|&nbsp; "
            f"<b>PEFT adapter:</b> <code>{adapter}</code> &nbsp;|&nbsp; "
            f"<b>Geo filter:</b> <code>{geo_status}</code> &nbsp;|&nbsp; "
            f"<b>Re-ranking:</b> <code>{rerank_status}</code>"
            f"</div>"
        )

    def reload(self, dataset: str, encoder: str, approach: str,
               color_mode: str = "rgb", use_lora: bool = False):
        """Switch this engine to another corpus/encoder in place (single-user use).
        The new state is built completely before it replaces the old one, so a failed
        switch keeps the previous corpus intact. The web UI does not call this: it
        gives each session its own configuration through an EnginePool."""
        try:
            cfg = settings_config(self.cfg, dataset, encoder, approach, color_mode, use_lora)
            fresh = type(self)(cfg)
            self.__dict__.update(fresh.__dict__)
            return _loaded_status(self.cfg, len(self.store)), self.stats_markdown()
        except Exception as exc:
            traceback.print_exc()
            return f"Error: {exc}", "<div class='stats-card stats-err'>Error</div>"

    def build_interface(self, pool: "Optional[EnginePool]" = None):
        import gradio as gr
        # `engine` (this startup engine) only seeds the widgets' initial values. Every
        # callback resolves the session's own engine from `pool`, keyed by the
        # configuration that session applied, so visitors never share a corpus switch.
        engine = self
        pool = pool or EnginePool(self)
        from src.encoders import list_encoders

        # Kept short so it doesn't dominate the first view; the dropdown choice
        # labels carry a one-line hint and the full explanation lives in About.
        APPROACH_HELP = "How each pair is scored. See **About** for what each option means."
        ENCODER_HELP = (
            "clip_vitl14 (general purpose, 768-d)  ·  georsclip (RS-pretrained ViT-B/32, 512-d)  "
            "·  remoteclip (RS-pretrained ViT-L/14, 768-d). Weights download on first use."
        )
        DATASET_HELP = (
            "Pair corpus to search over. Switching rebuilds (or loads cached) embeddings — "
            "press Apply Settings afterwards."
        )
        TOPK_HELP = "How many ranked change events to return."
        QUERY_HELP = ("Describe the land-cover change you are looking for. "
                      "Specific transitions (e.g. 'forest cleared to bare soil') work best.")
        COLOR_MODE_HELP = (
            "rgb = standard optical (R-G-B)  ·  nrg = NIR-Red-Green false colour "
            "(best zero-shot with GeoRSCLIP)  ·  ndvi = vegetation index (single channel × 3). "
            "NRG/NDVI need the NIR band and apply to Dynamic EarthNet only — other corpora "
            "are RGB-only (nrg/ndvi are greyed out and ignored). Changing requires Apply Settings."
        )
        LORA_HELP = (
            "Load LoRA-adapted embeddings pre-cached by run_pipeline --lora. "
            "Cache must exist for the selected encoder + color mode. "
            "Changing requires Apply Settings."
        )

        with gr.Blocks(title="Open Vocabulary Temporal Change Retrieval",
                       analytics_enabled=False) as demo:
            _fixture = "tests/fixtures" in str(self.cfg.root).replace("\\", "/")
            gr.Markdown(
                "# Open Vocabulary Temporal Change Retrieval\n"
                "Describe a land-cover change in plain language — the semantic change search "
                "engine finds the satellite image pairs (and the timestep) where it happened. "
                "*Research demo — results are approximate (see **About** below).*"
                + (f"\n\n**This hosted copy searches a bundled synthetic sample of {len(self.store)} image "
                   "pairs**, so its results only demonstrate the interface; run it locally with the real "
                   "corpora for real retrieval." if _fixture else "")
            )

            with gr.Accordion("About / How it works", open=False):
                gr.Markdown(
                    "**What it is.** A frozen vision–language model (CLIP / GeoRSCLIP / "
                    "RemoteCLIP) encodes each timestep of a bi-temporal satellite pair; the "
                    "**change** between them is matched against your text query by cosine "
                    "similarity. No per-query training — it is *open-vocabulary*.\n\n"
                    "**Open-vocabulary** means you can search for *any* change in free text — the "
                    "engine is never trained on the specific change types of a corpus. That is the "
                    "advantage over supervised models; the trade-off is lower absolute accuracy.\n\n"
                    "**Honest expectations.** This is a research demo, not a product. With frozen "
                    "encoders, open-vocabulary change retrieval is **weak in absolute terms** — the "
                    "best honestly-audited configuration reaches only **≈ 0.20 mAP** (5-fold "
                    "cross-validated), and recovery scales with how *visually salient* the change "
                    "is. Use the curated example queries for results that actually work.\n\n"
                    "**Approaches.** *Naive* = cos(text, After) — an image-retrieval baseline. "
                    "*Zero-shot* = cos(text, After) − cos(text, Before) — the temporal Δ, no "
                    "training. *Patch / localised* = top-3 per-patch Δ — it averages the three "
                    "most-changed patches, catching small localised change a whole-image embedding "
                    "would wash out; the headline configuration on Dynamic EarthNet. *PEFT adapter* = a "
                    "small trained projection head (loaded only if a matching adapter exists).\n\n"
                    "**On PEFT/LoRA (honest note).** Light fine-tuning was trained and evaluated on "
                    "Dynamic EarthNet and QFabric only. LoRA overfits the training scenes and collapses on "
                    "held-out ones; on Dynamic EarthNet the trained head is not significantly better than "
                    "frozen zero-shot under cross-validation, and nothing tested significantly beats frozen encoders + "
                    "NRG + patch scoring.\n\n"
                    "**Reading the results.** The **match score (0–1)** is the raw score min–max "
                    "normalised over the whole scored corpus (after any geographic filter): 1 is this "
                    "query's best pair and 0 its worst, a *relative* position, **not** a calibrated probability. The "
                    "**change heatmap** (jet) overlays the After image: warm = where the query's "
                    "presence grew most from Before→After; cool = little/no change. The **land-cover "
                    "change note** classes each result from its dataset label: *permanent* land-cover "
                    "change, *likely seasonal* (e.g. snow/ice, which recurs annually), or *stable* "
                    "(labelled weak/no change); *no label* on unlabelled corpora.\n\n"
                    "**Corpora (Settings → pick → Apply; each needs its data on disk, and the hosted "
                    "copy has only the bundled sample), sorted by best reported result:** LEVIR-CC "
                    "(building/road, the local default), SECOND-CC (six land-cover classes), "
                    "QFabric change-type (construction), Dynamic EarthNet (the report's primary "
                    "analysed corpus; subtle spectral change, low absolute results), and QFabric "
                    "construction-status (a distinct task with weak retrieval, included for "
                    "completeness). Switching reloads embeddings, so press **Apply Settings**.\n\n"
                    "**Note on QFabric here:** this is the reduced *2-date* TEOChatlas crop subset "
                    "(change-type / status retrieval only). The full *5-date* QFabric with polygon "
                    "change-masks (for temporal pinpointing + pixel localization) is a future-work "
                    "direction that was dropped from scope (the dataset source is access-gated) — it "
                    "is **not** the same as the 2-date subset shown here."
                )

            # ---- Query first — most important ----
            with gr.Row():
                q = gr.Textbox(
                    label="Change query", scale=5,
                    value="new buildings or houses constructed",
                    placeholder="e.g. a new road or street; a new water body",
                    info=QUERY_HELP,
                )
                a_dd = gr.Dropdown(
                    choices=[
                        ("Naive — match the After image", "naive"),
                        ("Zero-shot — temporal Δ (no training)", "zero_shot"),
                        ("Patch — localised Δ (best on DEN)", "patch"),
                        ("PEFT adapter — trained head", "peft"),
                    ],
                    value=engine.cfg.approach,
                    label="Approach",
                    info=APPROACH_HELP,
                    scale=2,
                )
                k = gr.Slider(1, _MAX_RESULTS, value=engine.cfg.top_k, step=1,
                              label="Top-K", info=TOPK_HELP, scale=1)
                go = gr.Button("Search", variant="primary", size="lg",
                               elem_id="search-btn")

            # NOTE: gr.Examples (Dataset component) triggers a frontend render loop in
            # gradio 6.14 when combined with the full component tree — page sticks on
            # "Loading..." and Firefox flags "slowing down". Plain fill-buttons give the
            # same click-to-fill UX without the Dataset component.
            gr.Markdown(
                "**Example queries — click to fill.** *Work best on the default LEVIR-CC corpus; "
                "see About for per-corpus tips.*"
            )
            with gr.Row():
                for _ex in [
                    "new buildings or houses constructed",
                    "a new road or street built",
                    "a new water body or flooding",
                    "trees or vegetation cleared or grown",
                ]:
                    gr.Button(_ex, size="sm").click(lambda v=_ex: v, None, q)

            # ---- Settings — one comprehensive menu (corpus · model · filters) ----
            _geo_available = engine._geo_filter is not None
            # only regions that hold a loaded pair: another region would filter every pair out
            _regions = (["All"] + sorted({r for r in (engine._geo_filter.region_of(p.location_id)
                                                        for p in engine.store.pairs) if r and r != "All"})
                        if _geo_available else ["All"])
            _rerank_available = engine._reranker is not None

            with gr.Accordion("Settings — corpus · model · filters", open=False):
                stats_md = gr.Markdown(engine.stats_markdown())

                gr.Markdown(
                    "**Corpus & model** — pick the dataset, encoder, colour mode or adapter, "
                    "then press **Apply settings** to (re)load embeddings."
                )
                with gr.Row():
                    d_dd = gr.Dropdown(_labeled(_app_dataset_choices(), DATASET_LABELS),
                                       value=engine.cfg.dataset,
                                       label="Dataset", info=DATASET_HELP)
                    e_dd = gr.Dropdown(_labeled(list_encoders(), ENCODER_LABELS),
                                       value=engine.cfg.encoder,
                                       label="Encoder", info=ENCODER_HELP)
                with gr.Row():
                    color_dd = gr.Dropdown(
                        ["rgb", "nrg", "ndvi"], value=engine.cfg.color_mode,
                        label="Color mode", info=COLOR_MODE_HELP,
                        # nrg/ndvi only apply to Dynamic EarthNet; greyed out otherwise.
                        interactive=(engine.cfg.dataset == "dynamic_earthnet"),
                    )
                    lora_chk = gr.Checkbox(
                        label="Use LoRA embeddings",
                        value=engine.cfg.use_lora,
                        info=LORA_HELP,
                    )
                with gr.Row():
                    apply = gr.Button("Apply settings", variant="secondary")
                    status = gr.Textbox(
                        label="Engine status", interactive=False, scale=4,
                        value=f"{engine.cfg.dataset} + {engine.cfg.encoder} | "
                              f"color={engine.cfg.color_mode} | "
                              f"{len(engine.store)} pairs | "
                              f"approach={engine.cfg.approach}")

                gr.Markdown(
                    "---\n**Filters & re-ranking** — applied on the **next Search** (no Apply "
                    "needed). Both require `aoi_metadata.json` in the dataset root (present for "
                    "Dynamic EarthNet); otherwise they are greyed out."
                )
                with gr.Row():
                    geo_chk = gr.Checkbox(
                        label="Geographic filter",
                        value=engine.cfg.geo_filter and _geo_available,
                        interactive=_geo_available,
                        info="Restrict results to one continental region."
                        if _geo_available
                        else "Requires aoi_metadata.json in the dataset root.",
                    )
                    geo_dd = gr.Dropdown(
                        _regions, value="All", label="Region",
                        interactive=_geo_available,
                    )
                with gr.Row():
                    rerank_chk = gr.Checkbox(
                        label="Re-rank results",
                        value=engine.cfg.rerank and _rerank_available,
                        interactive=_rerank_available,
                        info="Post-process ranking for spatial quality."
                        if _rerank_available
                        else "Requires aoi_metadata.json in the dataset root.",
                    )
                    rerank_dd = gr.Dropdown(
                        list(RERANK_STRATEGIES),
                        value=engine.cfg.rerank_strategy,
                        label="Strategy",
                        interactive=_rerank_available,
                        info=(
                            "diversity = prefer unique locations per result  ·  "
                            "coherence = cluster near the top-1 location"
                        ),
                    )

                # Colour mode only affects Dynamic EarthNet (the others have no NIR
                # band and pin rgb). Grey out nrg/ndvi for non-DEN corpora so the
                # control reflects what the engine actually does, instead of silently
                # falling back to rgb.
                def _color_for_dataset(dataset: str):
                    if dataset == "dynamic_earthnet":
                        return gr.update(interactive=True)
                    return gr.update(value="rgb", interactive=False)

                d_dd.change(_color_for_dataset, d_dd, color_dd)

                # This browser session's corpus configuration (Gradio copies the
                # initial value per session). Apply changes it for this session only.
                session_cfg = gr.State(engine.cfg)

                def apply_settings(dataset, encoder, approach, color_mode, use_lora, cfg):
                    new_cfg = settings_config(cfg, dataset, encoder, approach,
                                              color_mode, bool(use_lora))
                    try:
                        eng = pool.get(new_cfg)
                    except Exception as exc:
                        traceback.print_exc()
                        # The session keeps its previous, working configuration; the detail stays in the log.
                        return (f"Error: {type(exc).__name__} — the corpus could not be loaded with these settings.",
                                "<div class='stats-card stats-err'>Error</div>", cfg)
                    return _loaded_status(new_cfg, len(eng.store)), eng.stats_markdown(), new_cfg

                apply.click(apply_settings,
                            [d_dd, e_dd, a_dd, color_dd, lora_chk, session_cfg],
                            [status, stats_md, session_cfg])

            # ---- Results ----
            gr.Markdown("## Top match")
            # Holds the displayed top-K events server-side so a tile's "View"
            # button can swap that result into this detailed Top-match view.
            events_state = gr.State([])
            with gr.Group(elem_classes="top-card"):
                with gr.Row():
                    # Before/After as a draggable swipe comparison (the classic RS
                    # change-detection interaction) instead of static tiles — one
                    # wide component reads better and stacks cleanly on mobile.
                    # interactive=False = compare only, no upload.
                    # Square component (== tile aspect) + bounded-slider CSS so the
                    # swipe divider stays inside the image. buttons=[] -> no toolbar
                    # (download is handled by the explicit per-image buttons below).
                    cmp = gr.ImageSlider(
                        label="Before ↔ After — drag the divider to compare",
                        height=340, width=340, interactive=False, format="png",
                        buttons=[], elem_classes="bounded-slider",
                    )
                    # Heatmap as an After↔overlay swipe: drag to dial the change
                    # heatmap in and out over the After image (replaces a fixed-alpha
                    # static overlay — lets the user reveal exactly what changed).
                    hm_cmp = gr.ImageSlider(
                        label="After ↔ change heatmap — drag to reveal Δ",
                        height=340, width=340, interactive=False, format="png",
                        buttons=[], elem_classes="bounded-slider",
                    )
                gr.Markdown(
                    "_Left: swipe **Before ↔ After**. Right: swipe the **After** image against the "
                    "query-conditioned **change heatmap** (jet — warm = query-change grew, cool = "
                    "little/none) to dial the overlay in and out. See About._"
                )
                summary = gr.Markdown("*Press Search to retrieve.*")
                # Explicit, separately-named downloads for each image of the top
                # match — bigger, colour-coded, distinct boxes. Hidden until a
                # search/selection populates them.
                gr.Markdown("**Download images**")
                with gr.Row(elem_classes="dl-row"):
                    dl_before = gr.DownloadButton(
                        "Download Before", visible=False, size="lg",
                        elem_classes=["dl-btn", "dl-before"])
                    dl_after = gr.DownloadButton(
                        "Download After", visible=False, size="lg",
                        elem_classes=["dl-btn", "dl-after"])
                    dl_heat = gr.DownloadButton(
                        "Download heatmap", visible=False, size="lg",
                        elem_classes=["dl-btn", "dl-heat"])

            # ---- All matches (top-K) ----
            # A custom grid of full-size tiles, each with its own "View" button
            # directly beneath it (outside the image). Replaces gr.Gallery, which
            # clipped tiles behind an internal scrollbar and whose click was bound
            # to a hard-to-exit enlarge preview. MAX_RESULTS components are built up
            # front (== Top-K slider max) and shown/hidden per query.
            gr.Markdown("## All matches")
            MAX_RESULTS = _MAX_RESULTS  # == Top-K slider maximum
            tiles, view_btns = [], []
            with gr.Column(elem_classes="matches-grid"):
                for _r in range(2):
                    with gr.Row():
                        for _c in range(5):
                            with gr.Column(min_width=140):
                                tiles.append(gr.Image(
                                    visible=False, interactive=False, height=220,
                                    show_label=False, format="png", buttons=[],
                                    elem_classes="match-tile"))
                                view_btns.append(gr.Button(
                                    "View", visible=False, size="sm",
                                    elem_classes="view-btn"))

            with gr.Accordion("Results table", open=True):
                table = gr.Dataframe(
                    headers=["rank", "location", "Before", "After", "score",
                             "match (0–1)", "caption", "land cover change"],
                    interactive=False, wrap=True,
                )
                gr.Markdown(
                    "_Columns: **score** = raw change Δ-similarity (model units); "
                    "**match (0–1)** = relative rank within this query (min–max normalised, "
                    "not a probability); **caption** = the pair's dataset caption; "
                    "**land cover change** = permanent / likely-seasonal / stable note "
                    "(see About). Before/After are the two timesteps._"
                )

            # Export the ranked results so they can be saved / shared / analysed
            # offline. gr.File (not DownloadButton) serves the file reliably with
            # its real basename; hidden until a search produces rows.
            dl = gr.File(label="Download ranked results (CSV)", visible=False,
                         interactive=False)

            def _pill(c: float) -> str:
                cls = "low" if c < 0.4 else ("mid" if c < 0.7 else "")
                return (f"<span class='conf-pill {cls}' title='relative match score "
                        f"(min–max normalized over the candidate set), not a calibrated "
                        f"probability'>match {c:.2f}</span>")

            def _event_md(e) -> str:
                """The Top-match detail markdown for one change event."""
                return (
                    f"### {e.caption}\n"
                    f"{_pill(e.confidence)} &nbsp; "
                    f"**Location** `{e.location}` &nbsp; "
                    f"**Before → After** `{e.t1_key}` → `{e.t2_key}` &nbsp; "
                    f"**Score** `{e.score:.4f}`\n\n"
                    f"_Reasoning:_ {e.seasonal_note}."
                )

            def _dl(path):
                return gr.update(value=path, visible=path is not None)

            def _event_view(e, out_dir):
                """Materialise an event's images to named PNGs in *out_dir* and return
                the two swipe-slider tuples plus the three image paths (Before / After /
                heatmap). Named paths give meaningful download filenames; a slider
                tuple is None if a side is missing (None in a tuple breaks ImageSlider)."""
                b = materialize_image(e.t1_img, f"before_{e.location}_{e.t1_key}", out_dir)
                a = materialize_image(e.t2_img, f"after_{e.location}_{e.t2_key}", out_dir)
                h = materialize_image(
                    e.heatmap, f"heatmap_{e.location}_{e.t1_key}_to_{e.t2_key}", out_dir)
                before_after = (b, a) if b and a else None
                after_heat = (a, h) if a and h else None
                return before_after, after_heat, b, a, h

            def handle(text, approach, top_k,
                       geo_enabled, geo_region, rerank_enabled, rerank_strategy, cfg,
                       progress=gr.Progress()):
                try:
                    progress(0.05, desc="Scoring corpus against your query… "
                             "(first query on a dataset/approach encodes it — a few seconds)")
                    text = (text or "").strip()[:300]         # bounded, like the other inputs
                    if not text:
                        raise ValueError("empty query")
                    active_geo = geo_region if geo_enabled else "All"
                    active_rerank = rerank_strategy if rerank_enabled else None
                    top_k = clamp_top_k(top_k)
                    session_engine = pool.get(cfg)
                    evs = session_engine.query(
                        text, approach, top_k,
                        geo_region=active_geo,
                        rerank_strategy=active_rerank,
                    )
                except Exception as exc:
                    traceback.print_exc()
                    return (None, None, (
                        f"**Error:** {type(exc).__name__}.\n\n*If you just changed Dataset / Encoder / Color mode, "
                        "press **Apply Settings** first. For PEFT, an adapter must exist for the "
                        "selected encoder + colour mode.*"), [],
                        gr.update(visible=False), gr.update(visible=False),
                        gr.update(visible=False), gr.update(visible=False), [],
                        *[gr.update(value=None, visible=False) for _ in range(MAX_RESULTS)],
                        *[gr.update(visible=False) for _ in range(MAX_RESULTS)])
                if not evs:
                    return (None, None, (
                        "*No results for this query. Try a curated example above, or a different "
                        "**Approach** (patch / zero-shot).*"), [],
                        gr.update(visible=False), gr.update(visible=False),
                        gr.update(visible=False), gr.update(visible=False), [],
                        *[gr.update(value=None, visible=False) for _ in range(MAX_RESULTS)],
                        *[gr.update(visible=False) for _ in range(MAX_RESULTS)])
                progress(0.9, desc="Rendering results…")
                rows = [[e.rank, e.location, e.t1_key, e.t2_key,
                         round(e.score, 4), e.confidence, e.caption,
                         e.seasonal_note] for e in evs]
                # `shown` = events with a displayable image (the tile grid + View
                # buttons index these 1:1); `top` = the first displayable one, so the
                # Top-match panel and View #1 never point at different events.
                shown, top = displayable_events(evs)
                out_dir = _request_dir()
                tile_ups, btn_ups = [], []
                for i in range(MAX_RESULTS):
                    if i < len(shown):
                        e = shown[i]
                        kind = "heatmap" if e.heatmap is not None else "after"
                        tp = materialize_image(
                            e.heatmap or e.t2_img,
                            f"rank{e.rank}_{kind}_{e.location}_{e.t1_key}_to_{e.t2_key}",
                            out_dir)
                        tile_ups.append(gr.update(value=tp, visible=True))
                        btn_ups.append(gr.update(
                            value=f"View #{e.rank} · {e.location}", visible=True))
                    else:
                        tile_ups.append(gr.update(value=None, visible=False))
                        btn_ups.append(gr.update(visible=False))
                before_after, after_heat, b, a, h = _event_view(top, out_dir)
                csv_path = results_to_csv(rows, session_engine.cfg.dataset, out_dir)
                return (before_after, after_heat, _event_md(top), rows,
                        _dl(csv_path), _dl(b), _dl(a), _dl(h), shown,
                        *tile_ups, *btn_ups)

            def make_loader(i):
                """A tile's View button → load result *i* into the Top-match view and
                refresh the per-image download buttons (preview-free, always reliable)."""
                def _load(shown):
                    if not shown or i >= len(shown):
                        return tuple(gr.update() for _ in range(6))
                    e = shown[i]
                    before_after, after_heat, b, a, h = _event_view(e, _request_dir())
                    return before_after, after_heat, _event_md(e), _dl(b), _dl(a), _dl(h)
                return _load

            outputs = [cmp, hm_cmp, summary, table, dl,
                       dl_before, dl_after, dl_heat, events_state, *tiles, *view_btns]
            inputs = [q, a_dd, k, geo_chk, geo_dd, rerank_chk, rerank_dd, session_cfg]
            go.click(handle, inputs, outputs)
            q.submit(handle, inputs, outputs)
            for _i, _btn in enumerate(view_btns):
                _btn.click(make_loader(_i), [events_state],
                           [cmp, hm_cmp, summary, dl_before, dl_after, dl_heat])

            # Shareable deep links: ?q=<query>&approach=<naive|zero_shot|patch|peft>&k=<1-10>
            # prefill the controls on page load, so a search can be linked/bookmarked.
            # Values are validated; anything missing/invalid leaves that control as-is.
            _APPROACHES = {"naive", "zero_shot", "patch", "peft"}

            def _prefill_from_url(request: gr.Request = None):
                params = dict(request.query_params or {}) if request is not None else {}
                q_up = gr.update(value=params["q"]) if params.get("q") else gr.update()
                a_val = params.get("approach")
                a_up = gr.update(value=a_val) if a_val in _APPROACHES else gr.update()
                k_up = gr.update()
                try:
                    k_val = int(params.get("k", ""))
                    if 1 <= k_val <= _MAX_RESULTS:
                        k_up = gr.update(value=k_val)
                except (TypeError, ValueError):
                    pass
                return q_up, a_up, k_up

            demo.load(_prefill_from_url, None, [q, a_dd, k])
        return demo


class EnginePool:
    """Built engines shared by every browser session, keyed by what they load.

    A public Space serves all visitors from one process. Each session keeps its own
    RunConfig and looks its engine up here, so one visitor's "Apply settings" never
    changes another's corpus, and a search never reads an engine mid-rebuild: engines
    are not mutated once built, and a new configuration gets a new engine. Sessions
    on the same configuration share one engine. At most ``max_engines`` stay resident
    (least recently used out first), since each holds an encoder and embeddings; an
    evicted configuration is rebuilt (from the embedding cache) on its next use.
    """

    def __init__(self, first: "SemanticChangeSearch", max_engines: int = 2, factory=None):
        self._factory = factory or SemanticChangeSearch
        self._max = max(1, int(max_engines))
        self._lock = threading.Lock()        # guards the mapping
        self._build_lock = threading.Lock()  # one build at a time (memory-heavy)
        self._engines = OrderedDict([(_engine_key(first.cfg), first)])

    def _lookup(self, key):
        with self._lock:
            eng = self._engines.get(key)
            if eng is not None:
                self._engines.move_to_end(key)
            return eng

    def get(self, cfg: RunConfig) -> "SemanticChangeSearch":
        key = _engine_key(cfg)
        eng = self._lookup(key)
        if eng is not None:
            return eng
        with self._build_lock:
            eng = self._lookup(key)          # another session may have just built it
            if eng is not None:
                return eng
            eng = self._factory(cfg)         # raises on failure; nothing registered
            with self._lock:
                self._engines[key] = eng
                while len(self._engines) > self._max:
                    self._engines.popitem(last=False)
            return eng

    def __len__(self) -> int:
        with self._lock:
            return len(self._engines)


def parse_args():
    p = argparse.ArgumentParser(description="Semantic Change Search Engine")
    from src.encoders import list_encoders
    p.add_argument("--dataset", default="levir_mci",
                   choices=_app_dataset_choices())
    p.add_argument("--encoder", default="georsclip",
                   choices=list_encoders())
    p.add_argument("--approach", default="zero_shot",
                   choices=list(APPROACHES) + ["patch"])
    p.add_argument("--root", default=None,
                   help="dataset directory; default = the selected dataset's profile root")
    p.add_argument("--pairing", default="bimonthly")
    p.add_argument("--split", default=None,
                   help="split (train|val|test|all); default = the dataset's profile split")
    p.add_argument("--cache-dir", default=str(_PROJECT_ROOT / "data" / "cache"))
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--host", default="127.0.0.1",
                   help="Bind address. Default 127.0.0.1 (local only). Use 0.0.0.0 to "
                        "expose on the LAN; on a HuggingFace Space 0.0.0.0 is auto-selected.")
    # Spectral / encoder options. Default nrg + georsclip = the REPORT headline config.
    p.add_argument("--color-mode", default=None, choices=["rgb", "nrg", "ndvi"],
                   help="Image colour mode; default = the dataset's profile colour "
                        "(rgb for all but DEN, which defaults to nrg).")
    p.add_argument("--lora", action="store_true", default=False,
                   help="Load LoRA-adapted embeddings (must be pre-cached by run_pipeline --lora). "
                        "Toggle in-app via Settings.")
    p.add_argument("--no-lora", dest="lora", action="store_false")
    # Extension toggles
    p.add_argument("--geo-filter", action="store_true", default=False,
                   help="Enable geographic region filter at startup (toggle in-app anytime).")
    p.add_argument("--no-geo-filter", dest="geo_filter", action="store_false")
    p.add_argument("--rerank", action="store_true", default=False,
                   help="Enable post-retrieval re-ranking at startup (toggle in-app anytime).")
    p.add_argument("--no-rerank", dest="rerank", action="store_false")
    p.add_argument("--rerank-strategy", default="diversity",
                   choices=list(RERANK_STRATEGIES),
                   help="Re-ranking strategy: diversity (default) or coherence.")
    a = p.parse_args()
    # Resolve root / split / colour / loader-extras from the dataset's launch
    # profile when not explicitly overridden, so `python -m src.app` (default
    # dataset) loads from the right directory and the HF Space's explicit
    # --root/--dataset still win.
    prof = DATASET_PROFILES.get(a.dataset, {})
    root = a.root or prof.get("root") or str(_PROJECT_ROOT / "data" / "DynamicEarthNet")
    split = a.split if a.split is not None else prof.get("split", "test")
    color = a.color_mode or prof.get("color_mode") or "nrg"
    # Bind to localhost by default so the app + its error tracebacks are not exposed
    # on the LAN. A HuggingFace Space must bind 0.0.0.0 to be reachable, so detect it.
    on_space = bool(os.environ.get("SPACE_ID") or os.environ.get("SPACE_HOST"))
    server_name = "0.0.0.0" if on_space else a.host
    # Only surface full Python tracebacks in the browser when bound to localhost.
    show_error = server_name in ("127.0.0.1", "localhost")
    return RunConfig(
        dataset=a.dataset, encoder=a.encoder, approach=a.approach,
        root=root, pairing=a.pairing,
        split=None if split == "all" else split,
        cache_dir=a.cache_dir,
        color_mode=color,
        use_lora=a.lora,
        geo_filter=a.geo_filter,
        rerank=a.rerank,
        rerank_strategy=a.rerank_strategy,
        loader_extra=prof.get("loader_extra", {}),
    ), a.port, server_name, show_error


def main():
    import gradio as gr
    cfg, port, server_name, show_error = parse_args()
    print(f"Starting engine: dataset={cfg.dataset} encoder={cfg.encoder} "
          f"approach={cfg.approach}  bind={server_name}:{port}")
    engine = SemanticChangeSearch(cfg)
    demo = engine.build_interface()
    # Local system fonts only — no remote Google-font fetch (blocks render on slow net).
    theme = gr.themes.Ocean(
        # Bump the whole coordinated type scale up one notch (md -> lg) so every
        # font grows slightly while keeping the same relative ratios.
        text_size=gr.themes.sizes.text_lg,
        font=["system-ui", "-apple-system", "Segoe UI", "Arial", "sans-serif"],
        font_mono=["Consolas", "Monaco", "monospace"],
    )
    demo.launch(server_name=server_name, server_port=port, show_error=show_error,
                theme=theme, css=SemanticChangeSearch._CSS)


if __name__ == "__main__":
    main()
