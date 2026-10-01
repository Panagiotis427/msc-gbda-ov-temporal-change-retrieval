"""
Cross-validated DEN retrieval evaluation with confidence intervals.

Motivation (report §8.1): the committed DEN **test** split (110 pairs)
makes only **3** queries evaluable, so the headline 0.426 rests on n=3
and no generalisation across change-types can be claimed. This script removes
that single-split bottleneck two ways, **without re-encoding any imagery** — it
merges the cached train+val+test pair embeddings (= all 75 AOIs, 825 pairs):

1. **Full-corpus estimate.** Score every query over all 825 pairs → per-query AP
   with a **bootstrap 95% CI** (resampling pairs) and a **permutation p-value**
   vs random ranking. Pooling all AOIs makes 6 of the 10 queries evaluable under the
   default dominant-flip rule (vs 3 on test), and 8 under ``--relevance fraction``.
2. **K-fold AOI cross-validation.** Partition the 75 AOIs into K disjoint folds.
   - zero_shot: evaluate each fold independently → per-query AP across folds →
     mean ± CI (variance over AOI samples, not over a single 3-query split).
   - peft (``--peft``): for each fold, train the adapter on the *other* folds'
     pairs and evaluate on the held-out fold — a leakage-free cross-validated
     PEFT estimate (contrast with the train-fit 0.42/0.998 of report §8.2).

Note: with the corrected class mapping (see the README's notes on the report), four of
the ten queries have **zero positives anywhere in this 75-AOI subset** under the
dominant-flip rule (new buildings, urban expansion and the two agriculture/wetland
transitions); with ``--relevance fraction`` only the building and urban queries stay
empty. CV cannot conjure them; that needs more diverse data (report §11).

Run::

    python -m scripts.cv_eval --encoder georsclip --color-mode nrg --folds 5
    python -m scripts.cv_eval --encoder georsclip --color-mode rgb --folds 5 --peft
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.benchmark import _average_precision, encode_query
from src.datasets.dynamic_earthnet_pp import DENNpyDataset
from src.embeddings import PairEmbeddingStore, cache_path, cache_tag_for
from src.encoders import get_encoder
from src.queries import get_queries
from src.retrieval import ChangeRetriever
from src.stats import aoi_folds, bh_fdr, perm_p_value, rank_order

SPLITS = ("train", "val", "test")
BOOTSTRAP = 1000
PERM = 4000


def _std(x) -> float:
    """Sample standard deviation (ddof=1); 0.0 for fewer than two values."""
    a = np.asarray(x, dtype=np.float64)
    return float(np.std(a, ddof=1)) if a.size > 1 else 0.0


def _merge_stores(dataset_name, encoder_name, color, cache_dir):
    """Concatenate the cached train/val/test stores into one 75-AOI corpus store."""
    pairs, f1, f2, dim = [], [], [], None
    for sp in SPLITS:
        # Use the canonical cache-tag helper (single source of truth) rather
        # than a local re-implementation, so this can never drift from the
        # layout scripts.run_pipeline wrote the committed caches with.
        p = cache_path(cache_dir, dataset_name, encoder_name, tag=cache_tag_for(sp, color))
        if not p.exists():
            raise FileNotFoundError(f"missing cache {p} — run the pipeline for split={sp} first")
        st = PairEmbeddingStore.load(p)
        pairs.extend(st.pairs)
        f1.append(st.f_t1)
        f2.append(st.f_t2)
        dim = st.embed_dim
    return PairEmbeddingStore(dataset_name=dataset_name, encoder_name=encoder_name,
                              embed_dim=dim, pairs=pairs,
                              f_t1=np.concatenate(f1), f_t2=np.concatenate(f2))


def _sub_store(store, idx):
    return PairEmbeddingStore(dataset_name=store.dataset_name, encoder_name=store.encoder_name,
                              embed_dim=store.embed_dim, pairs=[store.pairs[i] for i in idx],
                              f_t1=store.f_t1[idx], f_t2=store.f_t2[idx])


def _rel_vector(dataset, pairs, predicate):
    return np.array([bool((lb := dataset.get_pair_label(p)) is not None and predicate(lb))
                     for p in pairs])


def _perm_p(rel_ranked_len, n_rel, obs_ap, rng, iters=PERM):
    """Unbiased one-sided permutation p-value P(random-ranking AP >= obs_ap) for a
    query with ``n_rel`` positives in ``N`` items, plus the mean random AP.

    The observed statistic is itself one realisation of the null, so the p-value is
    ``(#{draws >= obs} + 1) / (iters + 1)`` (``src.stats.perm_p_value``) — never an
    impossible 0.0.
    """
    N = rel_ranked_len
    draws = np.empty(iters)
    base = np.zeros(N, bool)
    base[:n_rel] = True
    for i in range(iters):
        perm = rng.permutation(N)
        rel = base[perm]
        hits = np.cumsum(rel)
        draws[i] = (hits / np.arange(1, N + 1))[rel].sum() / n_rel
    return perm_p_value(int(np.sum(draws >= obs_ap)), iters), float(draws.mean())


def main() -> None:
    ap = argparse.ArgumentParser(description="Cross-validated DEN retrieval eval with CIs")
    ap.add_argument("--encoder", default="georsclip")
    ap.add_argument("--color-mode", default="nrg", choices=["rgb", "nrg", "ndvi"])
    ap.add_argument("--root", default="data/DynamicEarthNet")
    ap.add_argument("--cache-dir", default="data/cache")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--approach", default="zero_shot", choices=["zero_shot", "naive"])
    ap.add_argument("--relevance", default="dominant", choices=["dominant", "fraction"],
                    help="dominant = committed dominant-class-flip queries; "
                         "fraction = pixel-fraction relevance (S1, see benchmark._gained/_lost)")
    ap.add_argument("--frac-thresh", type=float, default=0.05,
                    help="pixel-fraction threshold for --relevance fraction")
    ap.add_argument("--prompt-ensemble", action="store_true",
                    help="average the query text embedding over prompt templates "
                         "(benchmark.PROMPT_TEMPLATES) instead of a single phrasing")
    ap.add_argument("--peft", action="store_true", help="also run leakage-free k-fold PEFT")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    enc = get_encoder(args.encoder)
    ds = DENNpyDataset(root=args.root, split=None, color_mode=args.color_mode)
    store = _merge_stores("dynamic_earthnet", args.encoder, args.color_mode, args.cache_dir)
    assert len(store) == len(ds.list_pairs()), (len(store), len(ds.list_pairs()))
    pairs = store.pairs
    if args.relevance == "fraction":
        from src.queries.den import frac_queries
        queries = frac_queries(args.frac_thresh)
        print(f"relevance=fraction (thresh={args.frac_thresh}) — pixel-fraction predicates")
    else:
        queries = get_queries("dynamic_earthnet")
    aois = sorted({p.location_id for p in pairs})
    print(f"corpus: {len(pairs)} pairs, {len(aois)} AOIs, {len(queries)} queries registered")

    retr = ChangeRetriever(store, enc)
    rel_all = {q.text: _rel_vector(ds, pairs, q.predicate) for q in queries}
    evaluable = [q for q in queries if rel_all[q.text].sum() > 0]
    print(f"evaluable on full corpus: {len(evaluable)} queries"
          + (" | prompt-ensemble ON" if args.prompt_ensemble else ""))
    tvec = {q.text: encode_query(enc, q.text, ensemble=args.prompt_ensemble) for q in evaluable}

    # --- 1. full-corpus per-query AP + bootstrap CI + permutation p ----------
    full = []
    N = len(pairs)
    for q in evaluable:
        rel = rel_all[q.text]
        scores = retr.score_vec(tvec[q.text], approach=args.approach)
        ap_obs = _average_precision(rel[rank_order(scores, rel)])
        # bootstrap over pairs
        boot = np.empty(BOOTSTRAP)
        for b in range(BOOTSTRAP):
            samp = rng.integers(0, N, N)
            r = rel[samp]
            if r.sum() == 0:
                boot[b] = 0.0
                continue
            boot[b] = _average_precision(r[rank_order(scores[samp], r)])
        p, randmean = _perm_p(N, int(rel.sum()), ap_obs, rng)
        full.append({"query": q.text, "n_relevant": int(rel.sum()), "ap": round(ap_obs, 4),
                     "ci95": [round(float(np.percentile(boot, 2.5)), 4),
                              round(float(np.percentile(boot, 97.5)), 4)],
                     "rand_ap": round(randmean, 4), "perm_p": round(p, 4)})
    macro_full = round(float(np.mean([r["ap"] for r in full])), 4) if full else 0.0
    # Benjamini-Hochberg FDR q-values across this file's evaluable-query family,
    # so downstream reporting can cite genuine q-values rather than raw perm_p.
    if full:
        for r, qv in zip(full, bh_fdr([r["perm_p"] for r in full])):
            r["bh_fdr"] = round(float(qv), 4)

    # --- 2. K-fold AOI CV ----------------------------------------------------
    # Shared, seed-only partition helper (src.stats.aoi_folds): independent of the
    # bootstrap/permutation draws consumed above and IDENTICAL to patch_eval.py's,
    # so the two scripts' CV numbers are directly comparable.
    aoi_fold = aoi_folds(aois, args.folds, args.seed)
    fold_of_pair = np.array([aoi_fold[p.location_id] for p in pairs])

    def fold_idx(k):
        return np.where(fold_of_pair == k)[0]

    # zero_shot CV: per-query AP per fold
    cv_zs = {q.text: [] for q in evaluable}
    fold_macro = []
    for k in range(args.folds):
        idx = fold_idx(k)
        sub = _sub_store(store, idx)
        rsub = ChangeRetriever(sub, enc)
        aps_this = []
        for q in evaluable:
            rel = rel_all[q.text][idx]
            if rel.sum() == 0:
                continue
            sc = rsub.score_vec(tvec[q.text], approach=args.approach)
            ap_k = _average_precision(rel[rank_order(sc, rel)])
            cv_zs[q.text].append(ap_k)
            aps_this.append(ap_k)
        fold_macro.append(float(np.mean(aps_this)) if aps_this else 0.0)
    cv_zs_summary = {qt: {"mean": round(float(np.mean(v)), 4),
                          "std": round(_std(v), 4),
                          "n_folds": len(v)} for qt, v in cv_zs.items() if v}
    cv_macro_mean = round(float(np.mean(fold_macro)), 4)
    cv_macro_std = round(_std(fold_macro), 4)

    out = {
        "dataset": "dynamic_earthnet", "encoder": args.encoder,
        "color_mode": args.color_mode, "approach": args.approach,
        "relevance": args.relevance, "frac_thresh": args.frac_thresh,
        "prompt_ensemble": args.prompt_ensemble,
        "n_evaluable_queries": len(evaluable),
        "n_pairs": N, "n_aois": len(aois), "folds": args.folds,
        "full_corpus": {"macro_mAP": macro_full, "per_query": full},
        "kfold_zero_shot": {"macro_mAP_mean": cv_macro_mean, "macro_mAP_std": cv_macro_std,
                            "fold_macro": [round(x, 4) for x in fold_macro],
                            "per_query": cv_zs_summary},
    }

    # --- 3. leakage-free k-fold PEFT (optional) ------------------------------
    if args.peft:
        from src.train import TrainConfig, train_adapter
        key = {(p.location_id, p.t1_key, p.t2_key): i for i, p in enumerate(pairs)}
        peft_macro = []
        peft_pq = {q.text: [] for q in evaluable}
        for k in range(args.folds):
            test_idx = fold_idx(k)
            train_aois = [a for a in aois if aoi_fold[a] != k]
            ds_tr = DENNpyDataset(root=args.root, split=None,
                                  aoi_filter=train_aois, color_mode=args.color_mode)
            tr_pairs = ds_tr.list_pairs()
            tr_idx = np.array([key[(p.location_id, p.t1_key, p.t2_key)] for p in tr_pairs])
            store_tr = _sub_store(store, tr_idx)
            cfg = TrainConfig(mode="difference", epochs=args.epochs, seed=args.seed)
            adapter, _ = train_adapter(ds_tr, store_tr, enc, cfg, verbose=False)
            sub = _sub_store(store, test_idx)
            rsub = ChangeRetriever(sub, enc, feature_mode="difference")
            rsub.set_adapter(adapter, feature_mode="difference")
            aps_this = []
            for q in evaluable:
                rel = rel_all[q.text][test_idx]
                if rel.sum() == 0:
                    continue
                sc = rsub.score_vec(tvec[q.text], approach="peft")
                ap_k = _average_precision(rel[rank_order(sc, rel)])
                peft_pq[q.text].append(ap_k)
                aps_this.append(ap_k)
            peft_macro.append(float(np.mean(aps_this)) if aps_this else 0.0)
            print(f"  [peft fold {k}] macro mAP={peft_macro[-1]:.4f} (train {len(tr_pairs)} pairs)")
        out["kfold_peft"] = {
            "macro_mAP_mean": round(float(np.mean(peft_macro)), 4),
            "macro_mAP_std": round(_std(peft_macro), 4),
            "fold_macro": [round(x, 4) for x in peft_macro],
            "per_query": {qt: {"mean": round(float(np.mean(v)), 4),
                               "std": round(_std(v), 4), "n_folds": len(v)}
                          for qt, v in peft_pq.items() if v},
        }

    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    rel_tag = "" if args.relevance == "dominant" else f"__{args.relevance}"
    rel_tag += "__ens" if args.prompt_ensemble else ""
    op = (Path(args.results_dir) /
          f"cv_eval__{args.encoder}__{args.color_mode}__{args.approach}{rel_tag}.json")
    op.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nfull-corpus macro mAP = {macro_full} ({len(full)} queries)")
    print(f"k-fold zero_shot macro mAP = {cv_macro_mean} ± {cv_macro_std} (n={args.folds} folds)")
    if args.peft:
        print(f"k-fold PEFT macro mAP = {out['kfold_peft']['macro_mAP_mean']} "
              f"± {out['kfold_peft']['macro_mAP_std']}")
    print(f"wrote {op}")
    for r in full:
        print(f"  {r['ap']:.3f}  CI{r['ci95']}  p={r['perm_p']:.3f}  q={r.get('bh_fdr', float('nan')):.3f}"
              f"  n={r['n_relevant']:3d}  {r['query'][:46]}")


if __name__ == "__main__":
    main()
