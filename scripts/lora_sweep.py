"""
LoRA rank/epoch sweep for GeoRSCLIP + NRG (the best zero-shot config).

Baseline (rank=4, alpha=8, 20 epochs) gives test mAP 0.071 vs frozen zero-shot
0.426 (results/lora_sweep.txt; the 0.159 of the older tracked LoRA rows predates the
loss and target-module corrections) — LoRA memorises the training AOIs. This sweep asks whether more capacity (higher
rank) or longer training closes the gap. Each config trains a fresh LoRA, merges
into a fresh encoder, and benchmarks zero_shot on train + test **in memory** —
nothing is written to models/ or data/cache, so the committed rank-4 LoRA and
its caches are never clobbered.

Run::

    python -m scripts.lora_sweep --configs 4:20 8:20 16:20 8:40
"""
from __future__ import annotations

import argparse

from src.benchmark import run_benchmark
from src.datasets.registry import build_dataset
from src.embeddings import compute_pair_embeddings
from src.encoders import get_encoder
from src.lora_train import LoRAConfig, merge_lora_into_encoder, train_lora
from src.retrieval import ChangeRetriever


def _eval_config(rank, alpha, epochs, *, root, color, seed=42):
    """Train LoRA, merge into a fresh encoder, benchmark zero_shot on train+test."""
    enc = get_encoder("georsclip")  # fresh instance — merge mutates in place
    ds_train = build_dataset("dynamic_earthnet", root=root, pairing="bimonthly",
                             split="train", color_mode=color)
    cfg = LoRAConfig(rank=rank, alpha=alpha, epochs=epochs, seed=seed)
    visual_lora, _ = train_lora(ds_train, enc, cfg, verbose=False)
    merge_lora_into_encoder(enc, visual_lora)

    out = {}
    for split in ("train", "test"):
        ds = build_dataset("dynamic_earthnet", root=root, pairing="bimonthly",
                           split=split, color_mode=color)
        store = compute_pair_embeddings(ds, enc)          # in-memory, no cache
        rep = run_benchmark(ds, ChangeRetriever(store, enc), approach="zero_shot")
        out[split] = rep.mAP
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="LoRA rank/epoch sweep (georsclip+nrg)")
    ap.add_argument("--root", default="data/DynamicEarthNet")
    ap.add_argument("--color", default="nrg")
    ap.add_argument("--configs", nargs="+", default=["4:20", "8:20", "16:20", "8:40"],
                    help="rank:epochs pairs (alpha defaults to 2*rank)")
    args = ap.parse_args()

    rows = []
    for spec in args.configs:
        rank, epochs = (int(x) for x in spec.split(":"))
        alpha = 2 * rank
        print(f"\n=== LoRA rank={rank} alpha={alpha} epochs={epochs} ===")
        res = _eval_config(rank, alpha, epochs, root=args.root, color=args.color)
        rows.append((rank, alpha, epochs, res["train"], res["test"]))
        print(f"  train mAP={res['train']:.4f}  test mAP={res['test']:.4f}")

    print("\n============= LoRA SWEEP (georsclip + "
          f"{args.color}, zero_shot mAP) =============")
    print(f"{'rank':>4} {'alpha':>5} {'epochs':>6} {'train':>8} {'test':>8}")
    for r, a, e, tr, te in rows:
        print(f"{r:>4} {a:>5} {e:>6} {tr:>8.4f} {te:>8.4f}")
    print("frozen zero-shot reference: train 0.025  test 0.426")
    print("===============================================================")


if __name__ == "__main__":
    main()
