"""
PEFT training: fit the lightweight ``ProjectionHead`` adapter so that a
bi-temporal change feature Δf is pulled towards the CLIP-text embedding of a
natural-language description of that change. Backbones stay frozen — only the
~0.5M-param adapter trains ("parameter-efficient fine-tuning").

Supervision is the *weak caption* DEN derives from its LULC labels
(``DENDataset.text_caption_for_pair`` → e.g. "agriculture replaced by
impervious surface", "stable forest and other vegetation land cover"). Stable
pairs are kept so the model learns change-vs-no-change.

Loss: masked symmetric InfoNCE over an in-batch similarity matrix. Pairs that
share an identical caption are treated as mutual positives (DEN captions
repeat heavily), which avoids the false-negative problem of plain diagonal
InfoNCE.

Evaluation is the real label-grounded benchmark (``src.benchmark``), comparing
zero-shot vs the trained PEFT adapter — not a synthetic identity-diagonal — on
a split held out from training (``--eval-split``, default ``test``; the adapter
trains on ``--split``, default ``train``).

CLI:
    python -m src.train --dataset dynamic_earthnet --root data/DynamicEarthNet \
        --encoder clip_vitl14 --mode difference --epochs 40
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from src.datasets.base import TemporalDataset
from src.datasets.registry import build_dataset
from src.embeddings import PairEmbeddingStore, cache_tag_for, load_or_compute
from src.encoders import get_encoder
from src.model import ProjectionHead, adapter_path, create_projection_head, save_adapter
from src.retrieval import ChangeRetriever
from src.benchmark import run_benchmark


def build_caption_dataset(
    dataset: TemporalDataset,
    store: PairEmbeddingStore,
    mode: str = "difference",
) -> Tuple[np.ndarray, List[str]]:
    """Δf for every pair + its weak caption (positives *and* stable pairs)."""
    delta = store.change_features(mode=mode)
    captions = [_caption(dataset, p) for p in store.pairs]
    return delta.astype(np.float32), captions


def _caption(dataset: TemporalDataset, pair) -> str:
    fn = getattr(dataset, "text_caption_for_pair", None)
    if fn is not None:
        return fn(pair)
    lb = dataset.get_pair_label(pair)
    return lb.change_type if lb else "unknown land cover change"


def _masked_infonce(
    proj: torch.Tensor,      # [B, D]  adapter(Δf), pre-norm
    text: torch.Tensor,      # [B, D]  frozen text embeddings, pre-norm
    pos_mask: torch.Tensor,  # [B, B]  bool, True where caption_i == caption_j
    temperature: float = 0.07,
) -> torch.Tensor:
    a = F.normalize(proj, dim=-1)
    t = F.normalize(text, dim=-1)
    logits = a @ t.t() / temperature           # [B, B]

    def _dir(lg: torch.Tensor) -> torch.Tensor:
        log_prob = F.log_softmax(lg, dim=-1)
        # mean log-prob over the positive set for each row
        pos = (log_prob * pos_mask).sum(-1) / pos_mask.sum(-1).clamp_min(1)
        return -pos.mean()

    return 0.5 * (_dir(logits) + _dir(logits.t()))


@dataclass
class TrainConfig:
    mode: str = "difference"
    epochs: int = 40
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    hidden_dims: Tuple[int, ...] = (512, 256)
    dropout: float = 0.3
    temperature: float = 0.07
    seed: int = 42


def train_adapter(
    dataset: TemporalDataset,
    store: PairEmbeddingStore,
    encoder,
    cfg: TrainConfig = TrainConfig(),
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> Tuple[ProjectionHead, Dict]:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    delta, captions = build_caption_dataset(dataset, store, mode=cfg.mode)
    X = torch.from_numpy(delta).float().to(device)             # [N, in_dim]
    with torch.no_grad():
        T = torch.from_numpy(
            encoder.encode_text(captions).astype(np.float32)
        ).to(device)                                           # [N, D]

    # caption_i == caption_j positive mask
    uniq = {c: i for i, c in enumerate(sorted(set(captions)))}
    cid = torch.tensor([uniq[c] for c in captions], device=device)
    full_pos = (cid[:, None] == cid[None, :])

    in_dim = X.shape[1]
    out_dim = T.shape[1]
    adapter = create_projection_head(
        input_dim=in_dim, output_dim=out_dim,
        hidden_dims=cfg.hidden_dims, dropout_rate=cfg.dropout,
    ).to(device)

    opt = torch.optim.Adam(adapter.parameters(), lr=cfg.lr,
                           weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

    n = X.shape[0]
    history = {"loss": []}
    for ep in range(cfg.epochs):
        adapter.train()
        perm = torch.randperm(n, device=device)
        ep_loss, nb = 0.0, 0
        for s in range(0, n, cfg.batch_size):
            idx = perm[s:s + cfg.batch_size]
            if idx.numel() < 2:
                continue
            loss = _masked_infonce(
                adapter(X[idx]), T[idx],
                full_pos[idx][:, idx], cfg.temperature,
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += loss.item()
            nb += 1
        sched.step()
        avg = ep_loss / max(nb, 1)
        history["loss"].append(avg)
        if verbose and (ep == 0 or (ep + 1) % 10 == 0 or ep == cfg.epochs - 1):
            print(f"  epoch {ep + 1:3d}/{cfg.epochs}  loss={avg:.4f}")
    return adapter, history


def main() -> None:
    ap = argparse.ArgumentParser(description="Train PEFT change-retrieval adapter")
    ap.add_argument("--dataset", default="dynamic_earthnet")
    ap.add_argument("--root", default="data/DynamicEarthNet")
    ap.add_argument("--pairing", default="bimonthly",
                    choices=["bimonthly", "monthly", "seasonal-quartet"])
    ap.add_argument("--encoder", default="clip_vitl14")
    ap.add_argument("--mode", default="difference",
                    choices=["difference", "concatenate"])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--cache-dir", default="data/cache")
    ap.add_argument("--split", default="train",
                    help="split the adapter trains on (DEN preprocessed: train|val|test|all)")
    ap.add_argument("--eval-split", default="test",
                    help="split the Before/After benchmark scores; keep it disjoint from "
                         "--split, or the PEFT numbers measure train-set fit")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    enc = get_encoder(args.encoder)

    def load(split):
        ds = build_dataset(args.dataset, root=args.root, pairing=args.pairing,
                           split=None if split == "all" else split)
        # Key the embedding cache by split (this CLI is rgb-only) via the canonical
        # tag helper, matching scripts.run_pipeline. Without a tag this read/wrote
        # the un-split-tagged cache, so it bypassed the split-keyed caches and
        # different --split runs clobbered one shared file (the "test+rgb -> empty
        # tag" drift cache_tag_for exists to prevent).
        store = load_or_compute(ds, enc, cache_dir=args.cache_dir,
                                cache_tag=cache_tag_for(split, "rgb"))
        return ds, store

    ds, store = load(args.split)
    if args.eval_split == args.split:
        eval_ds, eval_store = ds, store
    else:
        eval_ds, eval_store = load(args.eval_split)
    if args.split in (args.eval_split, "all"):
        print(f"WARNING: training split '{args.split}' covers the evaluation split "
              f"'{args.eval_split}': the After (PEFT) table scores pairs the adapter "
              "was trained on — a train-set fit, not a held-out estimate. "
              "scripts.run_pipeline reports held-out numbers.")

    cfg = TrainConfig(mode=args.mode, epochs=args.epochs,
                      batch_size=args.batch_size, lr=args.lr)

    retr = ChangeRetriever(eval_store, enc, feature_mode=args.mode)
    print(f"\nBefore (zero-shot, {args.eval_split} split):")
    print(run_benchmark(eval_ds, retr, approach="zero_shot").to_table())

    print(f"\nTraining adapter ({args.mode}, {args.epochs} epochs)...")
    adapter, hist = train_adapter(ds, store, enc, cfg)

    # Adapter filename via the canonical helper (single source of truth shared
    # with run_pipeline / export_results); a non-default --mode adds its suffix.
    out = args.out or str(adapter_path(ds.name, enc.name, mode=args.mode))
    save_adapter(out, adapter, {
        "input_dim": adapter.input_dim, "output_dim": adapter.output_dim,
        "hidden_dims": list(cfg.hidden_dims), "dropout_rate": cfg.dropout,
        "feature_mode": args.mode, "encoder_name": enc.name,
        "dataset_name": ds.name,
    })
    print(f"Saved adapter -> {out}")

    retr.set_adapter(adapter, feature_mode=args.mode)
    print(f"\nAfter (PEFT, {args.eval_split} split):")
    print(run_benchmark(eval_ds, retr, approach="peft").to_table())


if __name__ == "__main__":
    main()
