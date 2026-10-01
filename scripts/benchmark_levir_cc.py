"""
LEVIR-CC open-vocabulary change-retrieval benchmark (report §8.5).

Encodes (or loads cached) per-pair embeddings for the LEVIR-CC test split and
scores the three caption-grounded queries (src/queries/levir_cc.py) under the
naive and zero-shot approaches, for each frozen encoder. Writes one JSON per
(encoder) to results/ so the §8.5 numbers are reproducible and traceable like
the DEN / QFabric results. The shared embed->score->JSON flow lives in
``scripts._caption_benchmark`` (also used by benchmark_second_cc).

Run::

    python -m scripts.benchmark_levir_cc --root data/_levir_mci/extracted/LEVIR-MCI-dataset

The default root is the LEVIR-MCI directory: LEVIR-MCI is a strict superset of
LEVIR-CC (identical 10,077 pairs + identical ``LevirCCcaptions.json``, plus change
masks), so the ``levir_cc`` images are not duplicated on disk — both ``levir_cc``
and ``levir_mci`` read the same A/B pairs from this one directory.
"""
from __future__ import annotations

from scripts._caption_benchmark import run_caption_benchmark

_DEFAULT_ROOT = "data/_levir_mci/extracted/LEVIR-MCI-dataset"
_ROOT_HELP = (
    "LEVIR-CC/MCI dir (LevirCCcaptions.json + images/{split}/{A,B}); "
    "defaults to the LEVIR-MCI superset so images aren't duplicated"
)


def main() -> None:
    run_caption_benchmark(
        "levir_cc", display_name="LEVIR-CC",
        default_root=_DEFAULT_ROOT, root_help=_ROOT_HELP,
    )


if __name__ == "__main__":
    main()
