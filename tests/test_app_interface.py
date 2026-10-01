"""
Headless smoke test of the Gradio wiring: build the real interface on the
synthetic DEN fixture (mock encoder, no browser) and drive its callbacks the way
two concurrent visitors would.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from src.app import EnginePool, RunConfig, SemanticChangeSearch

FIXTURE = Path("tests/fixtures/den_tiny")


@pytest.fixture(scope="module")
def ui():
    if not FIXTURE.exists():
        pytest.skip("fixture missing")
    from src.datasets.registry import get_dataset
    from src.embeddings import compute_pair_embeddings
    from test_app import MockEncoder

    ds = get_dataset("dynamic_earthnet", root=str(FIXTURE), pairing_strategy="bimonthly")
    enc = MockEncoder()
    store = compute_pair_embeddings(ds, enc)
    cfg = RunConfig(dataset="dynamic_earthnet", root=str(FIXTURE))
    first = SemanticChangeSearch.from_components(ds, enc, store, cfg)

    def factory(c):
        # Another configuration builds another engine over the same fixture data;
        # an unknown dataset fails the way the real registry does.
        if c.dataset != "dynamic_earthnet":
            raise ValueError(f"Unknown dataset: '{c.dataset}'")
        return SemanticChangeSearch.from_components(ds, enc, store, c)

    pool = EnginePool(first, factory=factory)
    demo = first.build_interface(pool=pool)
    fns = {getattr(f.fn, "__name__", ""): f.fn for f in demo.fns.values()}
    return first, pool, cfg, fns


QUERY = "deforestation forest cleared to bare soil"


def _search(fns, cfg, top_k=3):
    return fns["handle"](QUERY, "zero_shot", top_k, False, "All", False, "diversity", cfg)


def test_search_returns_rows_and_request_scoped_exports(ui):
    _, _, cfg, fns = ui
    first = _search(fns, cfg)
    second = _search(fns, cfg)
    rows = first[3]
    assert rows and rows[0][0] == 1
    csv_1, csv_2 = first[4]["value"], second[4]["value"]
    assert Path(csv_1).name == Path(csv_2).name == "change_results_dynamic_earthnet.csv"
    assert Path(csv_1).parent != Path(csv_2).parent      # no cross-request overwrite


def test_apply_changes_only_the_calling_session(ui):
    first, pool, cfg, fns = ui
    status, _, cfg_b = fns["apply_settings"]("dynamic_earthnet", "clip_vitl14",
                                             "zero_shot", "nrg", False, cfg)
    assert status.startswith("Loaded") and cfg_b.color_mode == "nrg"
    assert pool.get(cfg_b) is not first
    # Session A still searches its own engine, unchanged.
    assert pool.get(cfg) is first
    assert _search(fns, cfg)[3] and _search(fns, cfg_b)[3]


def test_failed_apply_keeps_the_session_configuration(ui):
    _, _, cfg, fns = ui
    status, card, kept = fns["apply_settings"]("no_such_dataset", "clip_vitl14",
                                               "zero_shot", "rgb", False, cfg)
    assert status.startswith("Error") and "stats-err" in card
    assert kept is cfg


def test_oversized_top_k_is_clamped(ui):
    _, _, cfg, fns = ui
    out = _search(fns, replace(cfg), top_k=10_000)
    assert len(out[3]) <= 10


def _handle(fns, cfg, text, approach="zero_shot"):
    return fns["handle"](text, approach, 3, False, "All", False, "diversity", cfg)


def test_blank_query_gets_a_polite_message_without_searching(ui):
    _, _, cfg, fns = ui
    for blank in ("", "   \n ", None):
        out = _handle(fns, cfg, blank)
        assert out[3] == [] and out[2].startswith("*Type a description")


def test_overlong_query_is_refused_with_its_length(ui):
    _, _, cfg, fns = ui
    out = _handle(fns, cfg, "a" * 301)
    assert out[3] == [] and "301 characters" in out[2] and "300" in out[2]
    assert _handle(fns, cfg, "a" * 300)[3]            # at the limit it still runs


def test_search_error_shows_no_exception_detail(ui):
    _, _, cfg, fns = ui
    out = _handle(fns, cfg, QUERY, approach="peft")   # the fixture has no adapter
    assert out[3] == [] and out[2].startswith("**Error:** RuntimeError.")
    assert "no adapter found" not in out[2]
