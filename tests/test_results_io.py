"""Tests for benchmark serialization (``to_dict``/``from_dict``) + ``results_io``.

Pure data-level: builds ``BenchmarkReport`` objects by hand (no encoder, no
network), so this stays in the fast offline suite.
"""
from __future__ import annotations

import json

import pytest

from src.benchmark import BenchmarkReport, QueryResult
from src import results_io


def _report() -> BenchmarkReport:
    qr_perm = QueryResult(
        text="deforestation, forest cleared to bare soil",
        category="permanent",
        n_relevant=7,
        recall_at_k={1: 0.0, 3: 0.14, 5: 0.28, 10: 0.42},
        ap=0.123,
        seasonal_drift_at_k={1: 1.0, 3: 0.66, 5: 0.5, 10: 0.3},
    )
    qr_seas = QueryResult(
        text="seasonal snow melting away",
        category="seasonal",
        n_relevant=3,
        recall_at_k={1: 1.0, 3: 1.0, 5: 1.0, 10: 1.0},
        ap=0.9,
        seasonal_drift_at_k={1: 0.0, 3: 0.0, 5: 0.0, 10: 0.0},
    )
    return BenchmarkReport(
        approach="zero_shot", encoder="clip_vitl14", dataset="dynamic_earthnet",
        n_pairs=605, per_query=[qr_perm, qr_seas],
    )


def test_query_result_round_trip():
    qr = _report().per_query[0]
    back = QueryResult.from_dict(qr.to_dict())
    assert back == qr  # dataclass eq; int keys restored


def test_report_round_trip():
    rep = _report()
    d = rep.to_dict(color_mode="nrg", split="test", lora=True)
    back = BenchmarkReport.from_dict(d)
    assert back.approach == rep.approach
    assert back.encoder == rep.encoder
    assert back.n_pairs == rep.n_pairs
    assert [q.text for q in back.per_query] == [q.text for q in rep.per_query]
    assert back.mAP == pytest.approx(rep.mAP)


def test_to_dict_macro_and_metadata():
    d = _report().to_dict(color_mode="nrg", split="test", lora=True)
    assert d["schema_version"] == 1
    assert d["color_mode"] == "nrg" and d["split"] == "test" and d["lora"] is True
    assert d["k_values"] == [1, 3, 5, 10]
    # macro mAP == mean of the two APs
    assert d["macro"]["mAP"] == pytest.approx((0.123 + 0.9) / 2)
    # seasonal drift macro = mean over PERMANENT queries only (one here)
    assert d["macro"]["seasonal_drift_at_k"]["1"] == pytest.approx(1.0)
    # K keys are strings for JSON stability
    assert set(d["macro"]["recall_at_k"]) == {"1", "3", "5", "10"}


def test_result_path_flag_combos(tmp_path):
    base = results_io.result_path(tmp_path, "dynamic_earthnet", "clip_vitl14", "train")
    assert base.name == "dynamic_earthnet__clip_vitl14__train__rgb__zero_shot.json"
    lora = results_io.result_path(tmp_path, "dynamic_earthnet", "georsclip",
                                  "test", color="nrg", approach="zero_shot", lora=True)
    assert lora.name == "dynamic_earthnet__georsclip__test__nrg__zero_shot__lora.json"
    peft = results_io.result_path(tmp_path, "dynamic_earthnet", "remoteclip",
                                  "val", approach="peft")
    assert peft.name == "dynamic_earthnet__remoteclip__val__rgb__peft.json"
    # mode only appears when not the default 'difference' -> no clobber of diff results
    diff = results_io.result_path(tmp_path, "dynamic_earthnet", "clip_vitl14",
                                  "train", approach="peft", mode="difference")
    concat = results_io.result_path(tmp_path, "dynamic_earthnet", "clip_vitl14",
                                    "train", approach="peft", mode="concatenate")
    assert diff.name == "dynamic_earthnet__clip_vitl14__train__rgb__peft.json"
    assert concat.name == "dynamic_earthnet__clip_vitl14__train__rgb__peft__concatenate.json"
    assert diff != concat


def test_write_report_idempotent(tmp_path):
    rep = _report()
    p = results_io.result_path(tmp_path, "dynamic_earthnet", "clip_vitl14", "test")
    results_io.write_report(rep, p, color_mode="rgb", split="test")
    first = p.read_bytes()
    results_io.write_report(rep, p, color_mode="rgb", split="test")
    assert p.read_bytes() == first  # byte-identical rewrite


def test_load_all_and_read(tmp_path):
    rep = _report()
    p = results_io.result_path(tmp_path, "dynamic_earthnet", "clip_vitl14", "test")
    results_io.write_report(rep, p, color_mode="rgb", split="test")
    recs = results_io.load_all(tmp_path)
    assert len(recs) == 1
    assert recs[0]["encoder"] == "clip_vitl14"
    assert results_io.load_all(tmp_path / "does_not_exist") == []


def test_load_all_skips_rerank(tmp_path):
    """``*rerank*`` files use a different (nested-strategies) schema and must not
    reach the macro CSV, where they flatten into a blank row."""
    rep = _report()
    p = results_io.result_path(tmp_path, "dynamic_earthnet", "clip_vitl14", "test")
    results_io.write_report(rep, p, color_mode="rgb", split="test")
    # drop a rerank sidecar and a cv_eval sidecar, both with no top-level 'macro'
    (tmp_path / "dynamic_earthnet__georsclip__test_nrg__zero_shot__rerank.json").write_text(
        json.dumps({"dataset": "dynamic_earthnet", "strategies": {"baseline": {}}}),
        encoding="utf-8",
    )
    (tmp_path / "cv_eval__georsclip__nrg__zero_shot.json").write_text(
        json.dumps({"dataset": "dynamic_earthnet", "full_corpus": {"macro_mAP": 0.04}}),
        encoding="utf-8",
    )
    (tmp_path / "patch_eval__georsclip__nrg__hybrid.json").write_text(
        json.dumps({"dataset": "dynamic_earthnet", "kfold": {"macro_mAP_mean": 0.16}}),
        encoding="utf-8",
    )
    recs = results_io.load_all(tmp_path)
    assert len(recs) == 1 and recs[0]["encoder"] == "clip_vitl14"


def test_write_macro_csv(tmp_path):
    rep = _report()
    rec = rep.to_dict(color_mode="rgb", split="test")
    csv_path = results_io.write_macro_csv([rec, rec], tmp_path / "macro.csv")
    lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0].startswith("dataset,encoder,split,color_mode,approach,lora")
    assert len(lines) == 3  # header + 2 rows
