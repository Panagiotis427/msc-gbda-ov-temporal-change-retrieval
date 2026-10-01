"""
LoRA fine-tuning of the visual encoder for temporal change retrieval.

Unlike the ProjectionHead adapter (which trains on pre-cached frozen embeddings),
LoRA modifies the visual encoder's **FFN weights** in-place. This requires loading
images on-the-fly during training and re-caching embeddings afterwards.

Encoder-agnostic
----------------
The trainer touches no encoder-family internals: each encoder advertises a
``lora_visual_spec()`` (``src.encoders.base.LoRAVisualSpec``) that supplies the
trainable visual module, its LoRA target-module names, the PIL→tensor preprocess,
and a unified ``forward`` returning shared-space embeddings. Supported families:
open_clip (``georsclip``, ``remoteclip``) and HF-transformers CLIP (``clip_vitl14``).
An encoder without ``lora_visual_spec()`` cannot be LoRA-trained and the trainer
fails with a clear message rather than an obscure ``AttributeError``.

Architecture
------------
- LoRA applied to the ViT FFN only: ``c_fc``/``c_proj`` (open_clip ResBlock MLP)
  or ``fc1``/``fc2`` (HF-CLIP encoder-layer MLP) — chosen per-encoder by the spec.
  The attention projections are NOT adapted: open_clip's attention is an
  ``nn.MultiheadAttention`` whose forward calls ``F.multi_head_attention_forward``
  and reads ``out_proj.weight`` / ``in_proj_weight`` as raw tensors — it never
  invokes ``out_proj.forward``, so a PEFT LoRA wrapper on ``out_proj`` would
  receive no gradient and be a silent no-op. Adapting attention would require a
  custom MHA wrapper; here we adapt the FFN only.
- Trainable params: ~369K for ViT-B-32 (~0.42% of the visual encoder).
- Text encoder stays fully frozen throughout.
- Loss: masked symmetric InfoNCE, identical to the ProjectionHead trainer
  (``src.train._masked_infonce``): every same-caption pair is a mutual positive
  (mean log-prob over the positive set), so DEN's heavily-repeated captions do
  not fight each other as negatives.
- After training: LoRA weights are merged into the base model via
  ``merge_and_unload()``, then embeddings are re-computed and cached.

CLI
---
    python -m src.lora_train --root data/DynamicEarthNet --encoder georsclip \\
        --split train --color-mode nrg --epochs 20

Use via ``scripts/run_pipeline.py --lora`` for the full cross-split evaluation.
"""
from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from src.datasets.base import TemporalDataset
from src.datasets.registry import build_dataset
from src.encoders import get_encoder


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class LoRAConfig:
    rank: int = 4
    alpha: int = 8
    dropout: float = 0.1
    # None => use the encoder's own LoRA target modules (``lora_visual_spec().target_modules``):
    # ``c_fc``/``c_proj`` for open_clip, ``fc1``/``fc2`` for HF-CLIP. Set explicitly only
    # to override. FFN only — see module docstring for why attention is left out.
    target_modules: Optional[List[str]] = None
    epochs: int = 20
    lr: float = 1e-4
    batch_size: int = 8
    seed: int = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_train_pairs(
    dataset: TemporalDataset,
) -> Tuple[List[Any], List[str]]:
    """Return (pairs, captions) lists aligned by index."""
    pairs = dataset.list_pairs()
    captions = [
        getattr(dataset, "text_caption_for_pair", lambda p: "land cover change")(p)
        for p in pairs
    ]
    return pairs, captions


def _infonce_loss(
    delta: torch.Tensor,
    text: torch.Tensor,
    pos_mask: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Masked symmetric InfoNCE — identical formulation to
    ``src.train._masked_infonce``.

    Every same-caption pair is a *mutual positive*: the loss is the mean log-prob
    over each row's positive set (``pos_mask``), averaged over both directions
    (delta->text and text->delta). All columns stay in the softmax denominator, so
    same-caption rows are NOT treated as negatives of one another — the bug the
    previous single-diagonal-target version had (it left same-caption positives in
    the denominator while targeting only the diagonal, making repeated DEN captions
    fight each other).
    """
    a = F.normalize(delta, dim=-1)
    t = F.normalize(text, dim=-1)
    logits = a @ t.t() / temperature           # [B, B]

    def _dir(lg: torch.Tensor) -> torch.Tensor:
        log_prob = F.log_softmax(lg, dim=-1)
        pos = (log_prob * pos_mask).sum(-1) / pos_mask.sum(-1).clamp_min(1)
        return -pos.mean()

    return 0.5 * (_dir(logits) + _dir(logits.t()))


def _require_lora_spec(encoder: Any):
    """Return the encoder's :class:`~src.encoders.base.LoRAVisualSpec`, or raise a
    clear error if the encoder does not support visual LoRA."""
    spec_fn = getattr(encoder, "lora_visual_spec", None)
    if spec_fn is None:
        raise TypeError(
            f"Encoder {getattr(encoder, 'name', type(encoder).__name__)!r} does not "
            "support visual LoRA (no lora_visual_spec()). Supported encoders: "
            "clip_vitl14, georsclip, remoteclip."
        )
    return spec_fn()


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_lora(
    dataset: TemporalDataset,
    encoder: Any,
    cfg: LoRAConfig = LoRAConfig(),
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> Tuple[Any, Dict]:
    """
    Apply LoRA to the encoder's visual tower (via ``encoder.lora_visual_spec()``)
    and train on ``dataset`` pairs.

    Returns
    -------
    visual_lora : peft.PeftModel
        Trained LoRA-wrapped visual module (NOT yet merged).
    history : dict
        Training log with per-epoch loss.
    """
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise ImportError("peft required: pip install peft") from exc

    spec = _require_lora_spec(encoder)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Keep the encoder's own device attribute in sync: spec.preprocess output is
    # moved to `device` below, and other encoder calls (encode_text) read
    # encoder.device, so an explicit `device` differing from construction time
    # would otherwise cause a mismatch.
    encoder.device = device

    # ---- Apply LoRA to the visual tower (encoder-family-agnostic) ----
    spec.to_device(device)
    lora_cfg = LoraConfig(
        r=cfg.rank,
        lora_alpha=cfg.alpha,
        target_modules=cfg.target_modules or spec.target_modules,
        lora_dropout=cfg.dropout,
        bias="none",
    )
    # The whole encoder is frozen at construction (requires_grad=False); peft's
    # get_peft_model then marks only the LoRA deltas trainable. The optimizer below
    # filters on requires_grad, so non-visual towers never receive updates.
    visual_lora = get_peft_model(spec.module, lora_cfg)
    visual_lora.to(device).train()
    if verbose:
        visual_lora.print_trainable_parameters()

    opt = torch.optim.Adam(
        filter(lambda p: p.requires_grad, visual_lora.parameters()),
        lr=cfg.lr,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

    pairs, captions = _build_train_pairs(dataset)
    if verbose:
        print(f"LoRA training: {len(pairs)} pairs, {cfg.epochs} epochs, "
              f"batch={cfg.batch_size}, rank={cfg.rank}, alpha={cfg.alpha}")

    # Pre-encode captions (text encoder frozen, no grad needed)
    with torch.no_grad():
        text_all = torch.from_numpy(
            encoder.encode_text(captions).astype(np.float32)
        ).to(device)

    caption_ids = {c: i for i, c in enumerate(sorted(set(captions)))}
    cid_all = torch.tensor([caption_ids[c] for c in captions], device=device)

    indices = list(range(len(pairs)))
    history: Dict[str, List[float]] = {"loss": []}

    for epoch in range(cfg.epochs):
        random.shuffle(indices)
        epoch_losses: List[float] = []

        for start in range(0, len(indices), cfg.batch_size):
            batch_idx = indices[start: start + cfg.batch_size]
            if len(batch_idx) < 2:
                continue

            px_t1_list, px_t2_list, text_batch, cid_batch = [], [], [], []
            for i in batch_idx:
                pair = pairs[i]
                try:
                    im1, im2 = dataset.load_pair_images(pair)
                except Exception:
                    continue
                px_t1_list.append(spec.preprocess(im1))   # [C, H, W]
                px_t2_list.append(spec.preprocess(im2))
                text_batch.append(text_all[i])
                cid_batch.append(cid_all[i])

            if len(px_t1_list) < 2:
                continue

            px_t1 = torch.stack(px_t1_list).to(device)   # [B, C, H, W]
            px_t2 = torch.stack(px_t2_list).to(device)
            T = torch.stack(text_batch)              # [B, D]
            cid = torch.stack(cid_batch)

            f1 = spec.forward(visual_lora, px_t1)
            f2 = spec.forward(visual_lora, px_t2)
            # Design choice: training matches the text to the *direction* of the change (cosine
            # with the normalised difference), whereas retrieval ranks pairs by the un-normalised
            # t·(f2 - f1), which also rewards a larger change magnitude.
            delta = F.normalize(f2 - f1, dim=-1)    # [B, D]

            pos_mask = (cid[:, None] == cid[None, :])
            loss = _infonce_loss(delta, T, pos_mask)

            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_losses.append(loss.item())

        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        history["loss"].append(mean_loss)
        sched.step()
        if verbose and (epoch % 5 == 0 or epoch == cfg.epochs - 1):
            print(f"  epoch {epoch+1:3d}/{cfg.epochs}  loss={mean_loss:.4f}")

    return visual_lora, history


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------

def save_lora(visual_lora: Any, path: str | Path) -> None:
    """Save LoRA adapter weights (not full model — only delta params)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    visual_lora.save_pretrained(str(path))


def merge_lora_into_encoder(encoder: Any, visual_lora: Any) -> None:
    """Merge trained LoRA weights into encoder in-place, unload peft wrapper."""
    merged = visual_lora.merge_and_unload()
    spec = _require_lora_spec(encoder)
    spec.set_module(merged)
    for p in merged.parameters():
        p.requires_grad = False


def load_lora_into_encoder(encoder: Any, path: str | Path) -> None:
    """Load a saved LoRA adapter from ``path`` and merge into ``encoder``."""
    try:
        from peft import PeftModel
    except ImportError as exc:
        raise ImportError("peft required: pip install peft") from exc

    spec = _require_lora_spec(encoder)
    visual_lora = PeftModel.from_pretrained(spec.module, str(path))
    merge_lora_into_encoder(encoder, visual_lora)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main() -> None:
    ap = argparse.ArgumentParser(description="Train LoRA adapter on visual encoder")
    ap.add_argument("--root", default="data/DynamicEarthNet")
    ap.add_argument("--dataset", default="dynamic_earthnet")
    ap.add_argument("--split", default="train")
    ap.add_argument("--encoder", default="georsclip",
                    choices=["clip_vitl14", "georsclip", "remoteclip"])
    ap.add_argument("--color-mode", default="nrg",
                    choices=["rgb", "nrg", "ndvi"])
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--alpha", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out", default=None,
                    help="Output dir for LoRA weights (default: models/<key>__lora/)")
    ap.add_argument("--cache-dir", default="data/cache")
    args = ap.parse_args()

    from src.embeddings import cache_tag_for, load_or_compute

    enc = get_encoder(args.encoder)
    # build_dataset (not get_dataset) so the dataset's opts-adapter runs — it drops
    # kwargs a loader can't accept and normalises split="all" → None, exactly like
    # every other pipeline entry point.
    ds = build_dataset(
        args.dataset,
        root=args.root,
        split=args.split,
        color_mode=args.color_mode,
    )

    cfg = LoRAConfig(
        rank=args.rank,
        alpha=args.alpha,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
    )
    visual_lora, history = train_lora(ds, enc, cfg, verbose=True)

    color_tag = f"_{args.color_mode}" if args.color_mode != "rgb" else ""
    out_dir = args.out or f"models/{ds.name}__{enc.name}{color_tag}__lora"
    save_lora(visual_lora, out_dir)
    print(f"LoRA weights saved → {out_dir}")

    print("Merging LoRA into encoder and re-computing embeddings ...")
    merge_lora_into_encoder(enc, visual_lora)
    # Canonical LoRA cache tag (split + colour + _lora suffix) via the single-source
    # helper, instead of re-deriving the string here (which bypassed its guard
    # assertions and would drift if the tag format changed).
    cache_tag = cache_tag_for(args.split, args.color_mode, lora=True)
    # force=True: the just-merged adapter changes embeddings without changing the
    # pair-set, so a stale LoRA cache must not be reused.
    store = load_or_compute(ds, enc, cache_dir=args.cache_dir, cache_tag=cache_tag, force=True)
    print(f"Re-cached {len(store.pairs)} pairs with LoRA-adapted encoder.")
    print(f"Cache tag: {cache_tag}")
    print(f"Final loss: {history['loss'][-1]:.4f}")


if __name__ == "__main__":
    _main()
