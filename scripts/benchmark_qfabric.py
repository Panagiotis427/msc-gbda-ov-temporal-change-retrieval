"""
Label-grounded QFabric benchmark (TEOChatlas crops, real change-type labels).

Encodes a stratified subset of QFabric before/after crops with each project
encoder and runs the change-type retrieval benchmark (naive + zero_shot),
writing results/qfabric_teo__<encoder>__eval__rgb__<approach>.json — the
quantitative second-dataset result (different taxonomy + temporal axis from DEN).

Optionally extracts the QFabric crops from the TEOChatlas eval tar first
(``--extract-from <tar.gz>``) — only ``*/QFabric/*`` members, into ``--crops-root``.

Run::

    # one-time extract (after scripts downloaded the 13.9 GB eval tar):
    python -m scripts.benchmark_qfabric --extract-from data/QFabric/_teochatlas/eval/TEOChatlas_images.tar.gz \\
        --crops-root data/QFabric/teochat_crops --extract-only

    # benchmark (encodes on GPU; 2,476 pairs x 3 encoders at --max-per-class 120):
    python -m scripts.benchmark_qfabric --crops-root data/QFabric/teochat_crops \\
        --max-per-class 120 --results-dir results
"""
from __future__ import annotations

import argparse
import tarfile
from pathlib import Path

from src.benchmark import run_benchmark
from src.datasets.registry import build_dataset
from src.embeddings import load_or_compute
from src.encoders import get_encoder
from src.model import adapter_path, save_adapter
from src.results_io import load_all, result_path, write_macro_csv, write_report
from src.retrieval import ChangeRetriever
from src.train import TrainConfig, train_adapter


def extract_qfabric(tar_path: str, crops_root: str) -> int:
    """Extract only ``QFabric/`` members from the TEOChatlas eval tar (flat)."""
    out = Path(crops_root)
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    with tarfile.open(tar_path, "r:gz") as tf:
        for m in tf:
            if not m.isfile() or "QFabric/" not in m.name or not m.name.endswith(".tif"):
                continue
            data = tf.extractfile(m)
            if data is None:
                continue
            (out / Path(m.name).name).write_bytes(data.read())
            n += 1
            if n % 2000 == 0:
                print(f"  extracted {n} QFabric crops...")
    print(f"Extracted {n} QFabric .tif crops -> {crops_root}")
    return n


def _run_zeroshot(args) -> list:
    """Whole-corpus naive/zero_shot benchmark (report §8.5: change-type /
    status-transition, selected by ``args.dataset``)."""
    written = []
    for enc_name in args.encoders:
        print(f"\n=== encoder: {enc_name} (zero-shot) ===")
        enc = get_encoder(enc_name)
        ds = build_dataset(args.dataset, root=args.crops_root, labels_path=args.labels,
                           max_per_class=args.max_per_class, seed=args.seed)
        store = load_or_compute(ds, enc, cache_dir=args.cache_dir,
                                cache_tag=f"{args.cache_prefix}eval_mpc{args.max_per_class}")
        retr = ChangeRetriever(store, enc)
        for appr in ("naive", "zero_shot"):
            rep = run_benchmark(ds, retr, approach=appr)
            p = result_path(args.results_dir, args.dataset, enc_name, "eval",
                            color="rgb", approach=appr)
            write_report(rep, p, color_mode="rgb", split="eval")
            written.append(p.name)
            print(f"  {appr:10s} mAP={rep.mAP:.4f}  N={rep.n_pairs}")
    return written


def _run_peft(args) -> list:
    """Train a ProjectionHead adapter on a held-out train split; evaluate
    naive/zero_shot/peft on train + test (report §8.5). Difference mode."""
    written = []
    mpc = args.max_per_class
    for enc_name in args.encoders:
        print(f"\n=== encoder: {enc_name} (PEFT, difference) ===")
        enc = get_encoder(enc_name)
        ds_tr = build_dataset(args.dataset, root=args.crops_root, labels_path=args.labels,
                              max_per_class=mpc, seed=args.seed, split="train")
        store_tr = load_or_compute(ds_tr, enc, cache_dir=args.cache_dir,
                                   cache_tag=f"{args.cache_prefix}train_mpc{mpc}")
        print(f"  training adapter on {len(store_tr)} train pairs ({args.epochs} ep)...")
        cfg = TrainConfig(mode="difference", epochs=args.epochs, seed=args.seed)
        adapter, _ = train_adapter(ds_tr, store_tr, enc, cfg, verbose=False)
        apath = str(adapter_path(args.dataset, enc_name))
        save_adapter(apath, adapter, {
            "input_dim": adapter.input_dim, "output_dim": adapter.output_dim,
            "hidden_dims": list(cfg.hidden_dims), "dropout_rate": cfg.dropout,
            "feature_mode": "difference", "encoder_name": enc_name,
            "dataset_name": args.dataset,
        })
        print(f"  saved adapter -> {apath}")
        for split in ("train", "test"):
            ds = build_dataset(args.dataset, root=args.crops_root, labels_path=args.labels,
                               max_per_class=mpc, seed=args.seed, split=split)
            store = load_or_compute(ds, enc, cache_dir=args.cache_dir,
                                    cache_tag=f"{args.cache_prefix}{split}_mpc{mpc}")
            retr = ChangeRetriever(store, enc, feature_mode="difference")
            for appr in ("naive", "zero_shot"):
                rep = run_benchmark(ds, retr, approach=appr)
                p = result_path(args.results_dir, args.dataset, enc_name, split,
                                color="rgb", approach=appr)
                write_report(rep, p, color_mode="rgb", split=split)
                written.append(p.name)
                print(f"  [{split:5s}] {appr:10s} mAP={rep.mAP:.4f}  N={rep.n_pairs}")
            retr.set_adapter(adapter, feature_mode="difference")
            rep = run_benchmark(ds, retr, approach="peft")
            p = result_path(args.results_dir, args.dataset, enc_name, split,
                            color="rgb", approach="peft")
            write_report(rep, p, color_mode="rgb", split=split)
            written.append(p.name)
            print(f"  [{split:5s}] {'peft':10s} mAP={rep.mAP:.4f}  N={rep.n_pairs}")
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description="QFabric change-type retrieval benchmark")
    ap.add_argument("--crops-root", default="data/QFabric/teochat_crops")
    ap.add_argument("--labels", default=None,
                    help="Label sidecar; defaults per --status (change-type vs status).")
    ap.add_argument("--status", action="store_true",
                    help="Benchmark RQA5 status-transition retrieval (dataset "
                         "'qfabric_status', report §8.5) instead of RQA2 change-type.")
    ap.add_argument("--encoders", nargs="+",
                    default=["clip_vitl14", "georsclip", "remoteclip"])
    ap.add_argument("--max-per-class", type=int, default=120,
                    help="Stratified cap per class (DEN-scale corpus).")
    ap.add_argument("--cache-dir", default="data/cache")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--extract-from", default=None,
                    help="TEOChatlas eval tar.gz to extract QFabric crops from first.")
    ap.add_argument("--extract-only", action="store_true")
    ap.add_argument("--peft", action="store_true",
                    help="Train a ProjectionHead adapter on a held-out train split and "
                         "evaluate naive/zero_shot/peft on train+test (report §8.5).")
    ap.add_argument("--epochs", type=int, default=40)
    args = ap.parse_args()

    # Resolve the dataset track: RQA2 change-type (default) vs RQA5 status-transition.
    args.dataset = "qfabric_status" if args.status else "qfabric_teo"
    args.cache_prefix = "status_" if args.status else ""
    if args.labels is None:
        args.labels = ("data/QFabric/qfabric_status_labels.json" if args.status
                       else "data/QFabric/qfabric_teo_labels.json")

    if args.extract_from:
        extract_qfabric(args.extract_from, args.crops_root)
        if args.extract_only:
            return

    written = (_run_peft(args) if args.peft else _run_zeroshot(args))

    if written:
        all_recs = load_all(args.results_dir)
        write_macro_csv(all_recs, f"{args.results_dir}/macro_summary.csv")
        print(f"\nWrote {len(written)} QFabric result(s); macro CSV now {len(all_recs)} rows.")


if __name__ == "__main__":
    main()
