"""
Track 4 --- bounded anti-memorization check: does feature-space augmentation
rescue the PEFT adapter out-of-distribution?

The report (§8.2) found the learned ProjectionHead adapter memorizes training AOIs
and collapses to <= frozen zero-shot on held-out AOIs. The open question (plan
Track 4 / future-work "PEFT anti-memorisation via augmentation") is whether a
regularizer that stops the adapter memorizing *exact* training Delta-f vectors
recovers any out-of-distribution signal.

Image augmentation is impossible here without re-encoding (training runs on cached
per-pair embeddings, not pixels), so the architecturally-correct test is
**embedding-space augmentation**: add Gaussian noise to the change feature Delta-f
each batch (a standard feature-space regularizer / "don't memorize the exact
vector" prior). This file does NOT edit the shared training pipeline; it imports
the shared pieces (``_masked_infonce``, ``build_caption_dataset``,
``create_projection_head``) and the shared leakage-free CV helpers from
``scripts.cv_eval`` (``_merge_stores``, ``_sub_store``, ``_average_precision``,
``_std``), and runs three controlled arms on the **same** AOI folds:

  1. frozen zero-shot      (no training)
  2. PEFT, no augmentation (noise_std = 0)  -- reproduces the report §8.2 collapse
  3. PEFT + feature noise  (noise_std > 0)  -- the anti-memorization arm

so the comparison is leakage-free and apples-to-apples. Reported as an honest
positive or negative, never a centerpiece.

Run::

    python -m scripts.peft_augment_eval --encoder georsclip --color-mode nrg --folds 5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.benchmark import encode_query
from src.datasets.dynamic_earthnet_pp import DENNpyDataset
from src.encoders import get_encoder
from src.model import create_projection_head
from src.queries.den import frac_queries
from src.retrieval import ChangeRetriever
from src.stats import aoi_folds, rank_order
from src.train import _masked_infonce, build_caption_dataset
from scripts.cv_eval import _average_precision, _merge_stores, _rel_vector, _std, _sub_store


def _train_noisy_adapter(ds_tr, store_tr, enc, *, epochs, noise_std, seed, device):
    """Train a ProjectionHead on a fold's Delta-f with optional Gaussian feature
    noise. noise_std is in units of the per-batch feature std (so it scales with
    the embedding magnitude). noise_std=0 reproduces the shared train_adapter."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    delta, captions = build_caption_dataset(ds_tr, store_tr, mode="difference")
    X = torch.from_numpy(delta).float().to(device)
    with torch.no_grad():
        T = torch.from_numpy(enc.encode_text(captions).astype(np.float32)).to(device)
    uniq = {c: i for i, c in enumerate(sorted(set(captions)))}
    cid = torch.tensor([uniq[c] for c in captions], device=device)
    full_pos = cid[:, None] == cid[None, :]

    adapter = create_projection_head(input_dim=X.shape[1], output_dim=T.shape[1],
                                     hidden_dims=(512, 256), dropout_rate=0.3).to(device)
    opt = torch.optim.Adam(adapter.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n = X.shape[0]
    for _ in range(epochs):
        adapter.train()
        perm = torch.randperm(n, device=device)
        for s in range(0, n, 32):
            idx = perm[s:s + 32]
            if idx.numel() < 2:
                continue
            xb = X[idx]
            if noise_std > 0:                       # feature-space augmentation
                sigma = noise_std * xb.std(dim=0, keepdim=True).clamp_min(1e-6)
                xb = xb + torch.randn_like(xb) * sigma
            loss = _masked_infonce(adapter(xb), T[idx], full_pos[idx][:, idx], 0.07)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    return adapter


def _cv_macro(store, pairs, aois, aoi_fold, folds, evaluable, rel_all, tvec, enc,
              *, arm, epochs, noise_std, seed, root, color_mode):
    """Leakage-free k-fold macro mAP for one arm. arm in {zero_shot, peft}."""
    key = {(p.location_id, p.t1_key, p.t2_key): i for i, p in enumerate(pairs)}
    fold_of = np.array([aoi_fold[p.location_id] for p in pairs])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fold_macro = []
    for k in range(folds):
        test_idx = np.where(fold_of == k)[0]
        sub = _sub_store(store, test_idx)
        rsub = ChangeRetriever(sub, enc, feature_mode="difference")
        approach = "zero_shot"
        if arm == "peft":
            train_aois = [a for a in aois if aoi_fold[a] != k]
            ds_tr = DENNpyDataset(root=root, split=None, aoi_filter=train_aois,
                                  color_mode=color_mode)
            tr_idx = np.array([key[(p.location_id, p.t1_key, p.t2_key)]
                               for p in ds_tr.list_pairs()])
            adapter = _train_noisy_adapter(ds_tr, _sub_store(store, tr_idx), enc,
                                           epochs=epochs, noise_std=noise_std,
                                           seed=seed, device=device)
            rsub.set_adapter(adapter, feature_mode="difference")
            approach = "peft"
        aps = []
        for q in evaluable:
            rel = rel_all[q.text][test_idx]
            if rel.sum() == 0:
                continue
            sc = rsub.score_vec(tvec[q.text], approach=approach)
            aps.append(_average_precision(rel[rank_order(sc, rel)]))
        fold_macro.append(float(np.mean(aps)) if aps else 0.0)
        print(f"  [{arm}{'' if noise_std==0 else f' noise={noise_std}'} fold {k}] "
              f"macro mAP={fold_macro[-1]:.4f}")
    return fold_macro


def main() -> None:
    ap = argparse.ArgumentParser(description="Track 4: feature-noise anti-memorization PEFT check")
    ap.add_argument("--encoder", default="georsclip")
    ap.add_argument("--color-mode", default="nrg", choices=["rgb", "nrg", "ndvi"])
    ap.add_argument("--root", default="data/DynamicEarthNet")
    ap.add_argument("--cache-dir", default="data/cache")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--frac-thresh", type=float, default=0.05)
    ap.add_argument("--noise-stds", type=float, nargs="+", default=[0.0, 0.25, 0.5, 1.0],
                    help="feature-noise levels to sweep (0 = no-aug baseline)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    enc = get_encoder(args.encoder)
    ds = DENNpyDataset(root=args.root, split=None, color_mode=args.color_mode)
    store = _merge_stores("dynamic_earthnet", args.encoder, args.color_mode, args.cache_dir)
    pairs = store.pairs
    aois = sorted({p.location_id for p in pairs})
    queries = frac_queries(args.frac_thresh)
    rel_all = {q.text: _rel_vector(ds, pairs, q.predicate) for q in queries}
    evaluable = [q for q in queries if rel_all[q.text].sum() > 0]
    tvec = {q.text: encode_query(enc, q.text) for q in evaluable}
    aoi_fold = aoi_folds(aois, args.folds, args.seed)
    print(f"corpus {len(pairs)} pairs / {len(aois)} AOIs / {len(evaluable)} evaluable queries "
          f"| {args.encoder} {args.color_mode}")

    common = dict(epochs=args.epochs, seed=args.seed, root=args.root,
                  color_mode=args.color_mode)

    print("zero-shot (frozen):")
    zs = _cv_macro(store, pairs, aois, aoi_fold, args.folds, evaluable, rel_all,
                   tvec, enc, arm="zero_shot", noise_std=0.0, **common)
    arms = {"zero_shot_frozen": {"cv_mean": round(float(np.mean(zs)), 4),
                                 "cv_std": round(_std(zs), 4)}}
    for ns in args.noise_stds:
        label = "peft_noaug" if ns == 0 else f"peft_noise_{ns}"
        print(f"{label}:")
        fm = _cv_macro(store, pairs, aois, aoi_fold, args.folds, evaluable, rel_all,
                       tvec, enc, arm="peft", noise_std=ns, **common)
        arms[label] = {"cv_mean": round(float(np.mean(fm)), 4),
                       "cv_std": round(_std(fm), 4), "noise_std": ns}

    out = {"dataset": "dynamic_earthnet", "encoder": args.encoder,
           "color_mode": args.color_mode, "relevance": "fraction",
           "frac_thresh": args.frac_thresh, "folds": args.folds,
           "epochs": args.epochs, "arms": arms}
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    op = Path(args.results_dir) / f"peft_augment__{args.encoder}__{args.color_mode}.json"
    op.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\n=== CV macro mAP (leakage-free 5-fold, fraction relevance) ===")
    for k, v in arms.items():
        print(f"  {k:18} {v['cv_mean']:.4f} ± {v['cv_std']:.4f}")
    print(f"wrote {op}")


if __name__ == "__main__":
    main()
