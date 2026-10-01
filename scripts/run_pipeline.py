"""
One-command reproducibility runner for the change-retrieval pipeline:

    data -> per-pair embeddings -> zero-shot benchmark -> PEFT train ->
    PEFT benchmark -> printed comparison table

Every heavy step is cached, so re-runs are fast. Use ``--encoder`` to repeat
for clip_vitl14 / georsclip / remoteclip and compare.

Examples
--------
Smoke (synthetic fixture, no downloads, ~seconds on CPU):
    python -m scripts.run_pipeline --root tests/fixtures/den_tiny --epochs 30

Real DEN subset, train-on-train / eval-on-test (default):
    python -m scripts.run_pipeline --root data/DynamicEarthNet \\
        --encoder clip_vitl14 --epochs 40

All splits evaluated independently:
    python -m scripts.run_pipeline --root data/DynamicEarthNet \\
        --encoder clip_vitl14 --eval-splits train val test --epochs 40

NRG false-colour (NIR-Red-Green) instead of RGB:
    python -m scripts.run_pipeline --root data/DynamicEarthNet \\
        --encoder clip_vitl14 --color-mode nrg --epochs 40

LoRA adapter instead of (or alongside) ProjectionHead:
    python -m scripts.run_pipeline --root data/DynamicEarthNet \\
        --encoder georsclip --color-mode nrg --lora --lora-epochs 20 \\
        --eval-splits train val test
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from src.datasets.registry import build_dataset
from src.embeddings import cache_tag_for, color_tag, load_or_compute
from src.encoders import get_encoder
from src.retrieval import ChangeRetriever
from src.benchmark import run_benchmark
from src.results_io import load_all, result_path, write_macro_csv, write_report
from src.train import TrainConfig, train_adapter
from src.model import adapter_path, save_adapter
from src.lora_train import LoRAConfig, train_lora, save_lora, merge_lora_into_encoder


def _build_ds(args, split, **extra):
    return build_dataset(
        args.dataset,
        root=args.root,
        pairing=args.pairing,
        split=split,
        color_mode=args.color_mode,
        **extra,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Reproducible change-retrieval pipeline")
    ap.add_argument("--dataset", default="dynamic_earthnet")
    ap.add_argument("--root", default="data/DynamicEarthNet")
    ap.add_argument("--pairing", default="bimonthly",
                    choices=["bimonthly", "monthly", "seasonal-quartet"])
    ap.add_argument("--encoder", default="clip_vitl14",
                    choices=["clip_vitl14", "georsclip", "remoteclip"])
    ap.add_argument("--mode", default="difference",
                    choices=["difference", "concatenate"])
    ap.add_argument("--color-mode", default="rgb",
                    choices=["rgb", "nrg", "ndvi"],
                    help="Image colour mode: rgb (default) | nrg (NIR-R-G) | ndvi")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--cache-dir", default="data/cache")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--lora", action="store_true",
                    help="Train a LoRA adapter on the visual encoder (re-caches embeddings).")
    ap.add_argument("--lora-epochs", type=int, default=20)
    ap.add_argument("--lora-rank", type=int, default=4)
    ap.add_argument("--lora-alpha", type=int, default=8)
    ap.add_argument("--train-split", default="train",
                    help="Split used to train the PEFT adapter (default: train)")
    ap.add_argument("--eval-splits", nargs="+", default=["test"],
                    help="Splits to evaluate on (default: test). Pass 'all' for all splits.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--results-dir", default=None,
                    help="If set, also dump each benchmark to "
                         "results/<...>.json + a macro_summary.csv (re-plottable).")
    ap.add_argument("--force", action="store_true",
                    help="Recompute embeddings even if a valid cache exists "
                         "(default = skip-if-done: reuse a valid cache and log the reuse).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the planned steps (encoder, splits, training) and exit "
                         "without computing anything.")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    eval_splits_plan = ["train", "val", "test"] if args.eval_splits == ["all"] else args.eval_splits
    if args.dry_run:
        steps = [f"embed+benchmark train split '{args.train_split}' "
                 f"(naive, zero_shot){'' if args.skip_train else ' + train PEFT adapter + peft benchmark'}"]
        if args.lora:
            steps.append(f"train LoRA (rank={args.lora_rank}, alpha={args.lora_alpha}, "
                         f"{args.lora_epochs} epochs) + re-cache + benchmark")
        for esplit in eval_splits_plan:
            steps.append(f"embed+benchmark eval split '{esplit}'")
        print(f"[dry-run] encoder={args.encoder}  dataset={args.dataset}  "
              f"mode={args.mode}  color={args.color_mode}  force={args.force}")
        for i, s in enumerate(steps, 1):
            print(f"[dry-run]  {i}. {s}")
        print("[dry-run] no embeddings/adapters computed or written.")
        return

    enc = get_encoder(args.encoder)
    color_suffix = color_tag(args.color_mode)
    # Tag artefacts trained on a non-default split so a `--train-split val` run never
    # overwrites the committed train-split adapter/LoRA (default 'train' -> no suffix,
    # back-compat with the committed `<ds>__<enc>[_<color>]__adapter.pt` names).
    split_tag = "" if args.train_split == "train" else f"_{args.train_split}"

    written: list = []

    def _emit(report, *, esplit, lora=False):
        """Print the table and, when --results-dir is set, persist JSON + collect
        the record for the macro CSV. Returns the report's mAP."""
        print(report.to_table())
        if args.results_dir:
            p = result_path(args.results_dir, args.dataset, enc.name, esplit,
                            color=args.color_mode, approach=report.approach,
                            lora=lora, mode=args.mode)
            write_report(report, p, color_mode=args.color_mode, split=esplit, lora=lora)
            written.append(report.to_dict(color_mode=args.color_mode,
                                          split=esplit, lora=lora))
        return report.mAP

    # ------------------------------------------------------------------
    # Training split: compute embeddings + train adapter
    # ------------------------------------------------------------------
    ds_train = _build_ds(args, split=args.train_split)
    store_train = load_or_compute(
        ds_train, enc,
        cache_dir=args.cache_dir,
        cache_tag=cache_tag_for(args.train_split, args.color_mode),
        force=args.force,
    )

    retr_train = ChangeRetriever(store_train, enc, feature_mode=args.mode)

    print(f"\n=== TRAINING SPLIT: {args.train_split.upper()} "
          f"({len(store_train)} pairs) ===")
    train_summary: dict[str, float] = {}
    for appr in ("naive", "zero_shot"):
        rep = run_benchmark(ds_train, retr_train, approach=appr)
        train_summary[appr] = _emit(rep, esplit=args.train_split)

    adapter = None
    if not args.skip_train:
        print(f"\nTraining PEFT adapter ({args.mode}, {args.epochs} epochs)...")
        cfg = TrainConfig(mode=args.mode, epochs=args.epochs, seed=args.seed)
        adapter, _ = train_adapter(ds_train, store_train, enc, cfg)
        apath = adapter_path(ds_train.name, enc.name, args.color_mode,
                             train_split=args.train_split, mode=args.mode)
        save_adapter(str(apath), adapter, {
            "input_dim": adapter.input_dim,
            "output_dim": adapter.output_dim,
            "hidden_dims": list(cfg.hidden_dims),
            "dropout_rate": cfg.dropout,
            "feature_mode": args.mode,
            "encoder_name": enc.name,
            "dataset_name": ds_train.name,
            "train_split": args.train_split,
            "color_mode": args.color_mode,
        })
        print(f"Saved adapter -> {apath}")
        retr_train.set_adapter(adapter, feature_mode=args.mode)
        rep = run_benchmark(ds_train, retr_train, approach="peft")
        train_summary["peft"] = _emit(rep, esplit=args.train_split)

    # ------------------------------------------------------------------
    # LoRA adapter (optional): fine-tune visual encoder, re-cache, benchmark
    # ------------------------------------------------------------------
    lora_store_train = None
    if args.lora:
        print(f"\nTraining LoRA adapter (rank={args.lora_rank}, alpha={args.lora_alpha}, "
              f"{args.lora_epochs} epochs) ...")
        lora_cfg = LoRAConfig(
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            epochs=args.lora_epochs,
            seed=args.seed,
        )
        visual_lora, lora_history = train_lora(ds_train, enc, lora_cfg, verbose=True)
        lora_dir = f"models/{ds_train.name}__{enc.name}{color_suffix}{split_tag}__lora"
        save_lora(visual_lora, lora_dir)
        print(f"Saved LoRA weights -> {lora_dir}/")

        # Merge LoRA into encoder and re-cache train embeddings
        merge_lora_into_encoder(enc, visual_lora)
        lora_cache_tag = cache_tag_for(args.train_split, args.color_mode, lora=True)
        # force=True: a freshly-trained adapter changes the embeddings even though
        # the pair-set is unchanged, so the LoRA cache MUST be recomputed — a stale
        # cache from a previous adapter would otherwise be silently reused.
        lora_store_train = load_or_compute(
            ds_train, enc,
            cache_dir=args.cache_dir,
            cache_tag=lora_cache_tag,
            force=True,
        )
        retr_lora_train = ChangeRetriever(lora_store_train, enc, feature_mode=args.mode)
        rep = run_benchmark(ds_train, retr_lora_train, approach="zero_shot")
        train_summary["lora"] = _emit(rep, esplit=args.train_split, lora=True)

    # ------------------------------------------------------------------
    # Evaluation splits
    # ------------------------------------------------------------------
    eval_splits = args.eval_splits
    if eval_splits == ["all"]:
        eval_splits = ["train", "val", "test"]

    eval_results: dict[str, dict[str, float]] = {}

    for esplit in eval_splits:
        if esplit == args.train_split and len(eval_splits) == 1:
            # Already benchmarked above; skip redundant re-run
            eval_results[esplit] = train_summary
            continue

        print(f"\n=== EVAL SPLIT: {esplit.upper()} ===")
        ds_eval = _build_ds(args, split=esplit if esplit != "all" else None)
        store_eval = load_or_compute(
            ds_eval, enc,
            cache_dir=args.cache_dir,
            cache_tag=cache_tag_for(esplit, args.color_mode),
            force=args.force,
        )
        retr_eval = ChangeRetriever(store_eval, enc, feature_mode=args.mode)

        split_summary: dict[str, float] = {}
        for appr in ("naive", "zero_shot"):
            rep = run_benchmark(ds_eval, retr_eval, approach=appr)
            split_summary[appr] = _emit(rep, esplit=esplit)

        if adapter is not None:
            retr_eval.set_adapter(adapter, feature_mode=args.mode)
            rep = run_benchmark(ds_eval, retr_eval, approach="peft")
            split_summary["peft"] = _emit(rep, esplit=esplit)

        if args.lora:
            lora_eval_tag = cache_tag_for(esplit, args.color_mode, lora=True)
            # force=True: recompute with the freshly-merged adapter (see above).
            lora_store_eval = load_or_compute(
                ds_eval, enc,
                cache_dir=args.cache_dir,
                cache_tag=lora_eval_tag,
                force=True,
            )
            retr_lora_eval = ChangeRetriever(lora_store_eval, enc, feature_mode=args.mode)
            rep = run_benchmark(ds_eval, retr_lora_eval, approach="zero_shot")
            split_summary["lora"] = _emit(rep, esplit=esplit, lora=True)

        eval_results[esplit] = split_summary

    # ------------------------------------------------------------------
    # Final comparison table
    # ------------------------------------------------------------------
    print("\n============= FINAL COMPARISON (mAP) =============")
    print(f"encoder={enc.name}  mode={args.mode}  "
          f"color={args.color_mode}  pairing={args.pairing}")

    approaches = sorted({k for v in eval_results.values() for k in v})
    header = f"{'split':10s}" + "".join(f"  {a:10s}" for a in approaches)
    print(header)
    print("-" * len(header))
    for sp, res in eval_results.items():
        row = f"{sp:10s}" + "".join(
            f"  {res.get(a, float('nan')):10.4f}" for a in approaches
        )
        print(row)
    print("===================================================")

    if args.results_dir and written:
        # Rebuild macro CSV from all results on disk so a subset run does not
        # clobber the full aggregate.
        all_recs = load_all(args.results_dir)
        csv_path = write_macro_csv(all_recs, f"{args.results_dir}/macro_summary.csv")
        print(f"\nWrote {len(written)} JSON result(s) this run -> {args.results_dir}/  "
              f"+ macro CSV ({len(all_recs)} rows) -> {csv_path}")


if __name__ == "__main__":
    main()
