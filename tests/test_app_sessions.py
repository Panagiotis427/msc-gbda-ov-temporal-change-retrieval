"""
The app serves every visitor from one process. These tests pin the guarantees
that keep visitors apart: each session's corpus lives in its own engine (a switch
by one never changes another's), a failed switch leaves the old corpus working,
exported files never collide across requests, and top-K is bounded server-side.
Headless and offline: fake engines or the synthetic DEN fixture, no CLIP.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest
from PIL import Image

from src import app as app_mod
from src.app import (EnginePool, RunConfig, SemanticChangeSearch, clamp_top_k,
                     materialize_image, results_to_csv, settings_config)


class _FakeEngine:
    """Stands in for SemanticChangeSearch: records what was built."""
    built: list = []

    def __init__(self, cfg: RunConfig):
        self.cfg = cfg
        _FakeEngine.built.append(cfg.dataset)


def _pool(max_engines=2, factory=_FakeEngine):
    _FakeEngine.built = []
    first = _FakeEngine(RunConfig(dataset="first"))
    _FakeEngine.built = []
    return first, EnginePool(first, max_engines=max_engines, factory=factory)


def test_sessions_with_different_settings_get_different_engines():
    first, pool = _pool()
    other = pool.get(RunConfig(dataset="other"))
    assert other is not first and other.cfg.dataset == "other"
    # The first session's engine is untouched by the second session's switch.
    assert pool.get(RunConfig(dataset="first")) is first
    assert first.cfg.dataset == "first"


def test_same_settings_share_one_engine_and_query_options_do_not_rebuild():
    first, pool = _pool()
    same = RunConfig(dataset="first", approach="patch", top_k=9, rerank=True)
    assert pool.get(same) is first
    assert _FakeEngine.built == []


def test_least_recently_used_engine_is_evicted_and_rebuilt_on_demand():
    first, pool = _pool(max_engines=2)
    pool.get(RunConfig(dataset="b"))
    pool.get(RunConfig(dataset="c"))            # evicts "first", the least recent
    assert len(pool) == 2
    rebuilt = pool.get(RunConfig(dataset="first"))
    assert rebuilt is not first and rebuilt.cfg.dataset == "first"
    assert _FakeEngine.built == ["b", "c", "first"]


def test_failed_build_registers_nothing_and_keeps_existing_engines():
    def boom(cfg):
        raise RuntimeError("corpus files missing")

    first, pool = _pool(factory=boom)
    with pytest.raises(RuntimeError, match="corpus files missing"):
        pool.get(RunConfig(dataset="broken"))
    assert len(pool) == 1
    assert pool.get(RunConfig(dataset="first")) is first


def test_concurrent_requests_for_a_new_config_build_it_once():
    def slow(cfg):
        time.sleep(0.05)
        return _FakeEngine(cfg)

    first, pool = _pool(factory=slow)
    got = []
    threads = [threading.Thread(target=lambda: got.append(pool.get(RunConfig(dataset="new"))))
               for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert _FakeEngine.built == ["new"]
    assert len({id(e) for e in got}) == 1


def test_settings_config_keeps_the_current_root_when_a_profile_dir_is_absent():
    base = RunConfig(dataset="dynamic_earthnet", root="tests/fixtures/den_tiny",
                     color_mode="nrg", cache_dir="data/cache")
    cfg = settings_config(base, "dynamic_earthnet", "clip_vitl14", "patch", "ndvi", True)
    assert cfg.color_mode == "ndvi" and cfg.use_lora and cfg.approach == "patch"
    assert cfg.cache_dir == "data/cache"
    # LEVIR pins rgb whatever the dropdown says.
    assert settings_config(base, "levir_mci", "georsclip", "zero_shot", "nrg").color_mode == "rgb"


def test_top_k_is_clamped_server_side():
    assert clamp_top_k(0) == 1
    assert clamp_top_k(5) == 5
    assert clamp_top_k("3") == 3
    assert clamp_top_k(100_000) == 10


def test_each_request_exports_into_its_own_directory():
    one, two = app_mod._request_dir(), app_mod._request_dir()
    assert one != two
    a = materialize_image(Image.new("RGB", (4, 4), "red"), "heatmap_loc_t1_to_t2", one)
    b = materialize_image(Image.new("RGB", (4, 4), "blue"), "heatmap_loc_t1_to_t2", two)
    assert Path(a).name == Path(b).name == "heatmap_loc_t1_to_t2.png"
    with Image.open(a) as img_a, Image.open(b) as img_b:
        assert img_a.getpixel((0, 0)) != img_b.getpixel((0, 0))
    c1 = results_to_csv([[1, "loc"]], "levir_mci", one)
    c2 = results_to_csv([[2, "loc"]], "levir_mci", two)
    assert c1 != c2 and Path(c1).read_text() != Path(c2).read_text()


def test_old_request_directories_are_pruned_but_never_the_new_one():
    made = [app_mod._request_dir() for _ in range(app_mod._REQUEST_DIRS_KEPT + 5)]
    live = [p for p in Path(app_mod._export_dir()).iterdir() if p.name.startswith("req_")]
    assert len(live) <= app_mod._REQUEST_DIRS_KEPT
    assert Path(made[-1]).is_dir()


FIXTURE = Path("tests/fixtures/den_tiny")


def test_failed_reload_keeps_the_previous_corpus():
    if not FIXTURE.exists():
        pytest.skip("fixture missing")
    from src.datasets.registry import get_dataset
    from src.embeddings import compute_pair_embeddings
    from _mocks import MockEncoderBase

    class Enc(MockEncoderBase):
        def encode_text(self, texts, batch_size=32):
            raise NotImplementedError

    ds = get_dataset("dynamic_earthnet", root=str(FIXTURE), pairing_strategy="bimonthly")
    enc = Enc()
    store = compute_pair_embeddings(ds, enc)
    engine = SemanticChangeSearch.from_components(
        ds, enc, store, RunConfig(dataset="dynamic_earthnet", root=str(FIXTURE)))
    status, _ = engine.reload("no_such_dataset", "clip_vitl14", "zero_shot")
    assert status.startswith("Error") and "no_such_dataset" not in status   # detail stays in the log
    assert engine.cfg.dataset == "dynamic_earthnet"
    assert engine.store is store and engine.dataset is ds
