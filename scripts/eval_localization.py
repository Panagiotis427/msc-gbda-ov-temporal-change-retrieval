"""
Track 3 --- quantitative change localization on LEVIR-MCI masks.

The brief asks for "a heatmap highlighting the specific spatial region of the
change". So far that deliverable was only qualitative (a colour overlay in the
Gradio app). LEVIR-MCI adds pixel-level building/road change masks on the same
pairs LEVIR-CC already scores, so the query-conditioned change heatmap can be
scored against ground truth.

For a pair with per-patch embeddings ``P1, P2`` (``[n_patch, D]``, L2-normed) and
a query vector ``t``, the change heatmap is the per-patch Δ-similarity
``delta_p = cos(t, P2_p) - cos(t, P1_p)`` (the same signal as the S3 patch scorer,
REPORT Appendix B.10, and ``src/heatmap.generate_change_heatmap``). The ground
truth is the LEVIR-MCI mask for the query's class (building or road),
area-downsampled to the encoder's patch grid. Two honest metrics, each scored
only on pairs that actually contain that class of change and have at least one
positive patch at grid resolution:

* ``pointing_game``: is the single highest-Δ patch a true change patch? Compared
  against the random-patch rate (the mean positive-patch fraction).
* ``patch_AP``:      average precision of the Δ map over the patch grid, macro-
  averaged over pairs. Compared against the same prevalence floor.

Only building and road carry masks in LEVIR-MCI; demolition / vegetation / water
have captions but no change mask, so localization is reported for the two masked
classes only (stated honestly). DEN has no instance masks at all, so this metric
is LEVIR-MCI-specific by design.

Run::

    python -m scripts.eval_localization --root data/_levir_mci/extracted/LEVIR-MCI-dataset
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - tqdm is a dep, but stay graceful
    def tqdm(x, **_):
        return x

from src.benchmark import encode_query
from src.datasets.registry import build_dataset
from src.encoders import get_encoder

# Per-dataset config: default extracted root + the module exposing QUERY_TO_MASK_CLASS.
# Both loaders share the same mask interface (has_mask, load_change_mask(pair, cls)).
_DATASETS = {
    "levir_mci": {
        "root": "data/_levir_mci/extracted/LEVIR-MCI-dataset",
        "mask_map": "src.datasets.levir_mci",
    },
    "second_cc": {
        "root": "data/_second_cc/extracted/SECOND-CC-AUG",
        "mask_map": "src.datasets.second_cc",
    },
}


def _patch_embeddings(ds, enc, enc_name, split, cache_dir, dataset):
    """Encode (or load) per-pair T1/T2 patch embeddings → P1, P2 [N, n_patch, D]."""
    from src.embeddings import patch_version_suffix
    cache = Path(cache_dir) / f"locpatch__{dataset}__{enc_name}__{split}{patch_version_suffix(enc_name)}.npz"
    pairs = ds.list_pairs()
    if cache.exists():
        d = np.load(cache, allow_pickle=False)
        if int(d["n"]) == len(pairs):
            print(f"loaded patch cache {cache} (N={int(d['n'])})")
            return d["p1"], d["p2"], pairs
    print(f"encoding {len(pairs)} pairs to patch embeddings ({enc_name}, {split})...")
    # Batch the encode: collect a chunk of tiles and encode each side in one pass
    # rather than one single-image forward per tile. Per-image patch tokens are
    # independent so the result is numerically identical; chunking bounds host
    # memory (mirrors src.embeddings.compute_patch_embeddings). Cold-cache only.
    p1, p2 = [], []
    batch = 32
    for s in tqdm(range(0, len(pairs), batch), desc=f"{enc_name} patches", unit="batch"):
        imgs1, imgs2 = [], []
        for pk in pairs[s:s + batch]:
            im1, im2 = ds.load_pair_images(pk)
            imgs1.append(im1)
            imgs2.append(im2)
        a = enc.encode_image_patches(imgs1, batch_size=batch)
        b = enc.encode_image_patches(imgs2, batch_size=batch)
        if a is None or b is None:
            raise RuntimeError(f"{enc_name} does not expose patch tokens")
        p1.append(np.asarray(a))
        p2.append(np.asarray(b))
    p1, p2 = np.concatenate(p1, axis=0), np.concatenate(p2, axis=0)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, p1=p1, p2=p2, n=len(pairs))
    print(f"  cached {cache}")
    return p1, p2, pairs


def _gt_patches(mask_bool: np.ndarray, side: int, pos_thresh: float) -> np.ndarray:
    """Area-downsample a full-res boolean mask to a [side*side] patch label vector.

    A patch is positive if the fraction of changed pixels falling in it exceeds
    ``pos_thresh`` (``0.0`` = any changed pixel in the cell)."""
    frac = cv2.resize(mask_bool.astype(np.float32), (side, side),
                      interpolation=cv2.INTER_AREA)
    return (frac > pos_thresh).reshape(-1)


def _average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    """AP of a ranking of patches by ``scores`` against boolean ``labels``."""
    order = np.argsort(-scores)
    lab = labels[order].astype(np.float32)
    csum = np.cumsum(lab)
    ranks = np.arange(1, len(lab) + 1)
    precision_at_hits = (csum / ranks)[lab > 0]
    return float(precision_at_hits.mean()) if precision_at_hits.size else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description="Change-localization eval (heatmap vs mask)")
    ap.add_argument("--dataset", default="levir_mci", choices=sorted(_DATASETS),
                    help="masked dataset to score (levir_mci building/road, "
                         "second_cc six land-cover classes)")
    ap.add_argument("--root", default=None,
                    help="dataset dir (defaults to the dataset's standard extracted root)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--encoders", nargs="+",
                    default=["georsclip", "clip_vitl14", "remoteclip"])
    ap.add_argument("--cache-dir", default="data/cache")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--pos-thresh", type=float, default=0.0,
                    help="patch is GT-positive if changed-pixel fraction > this")
    args = ap.parse_args()

    cfg = _DATASETS[args.dataset]
    root = args.root or cfg["root"]
    import importlib
    query_to_mask_class = importlib.import_module(cfg["mask_map"]).QUERY_TO_MASK_CLASS

    ds = build_dataset(args.dataset, root=root, split=args.split)
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)

    for enc_name in args.encoders:
        enc = get_encoder(enc_name)
        P1, P2, pairs = _patch_embeddings(ds, enc, enc_name, args.split,
                                          args.cache_dir, args.dataset)
        n_patch = P1.shape[1]
        side = int(round(n_patch ** 0.5))
        if side * side != n_patch:           # _gt_patches assumes a square grid
            raise ValueError(f"{enc_name}: non-square patch grid ({n_patch} patches); "
                             "the localization metric needs a square grid")
        out = {"dataset": args.dataset, "encoder": enc_name, "split": args.split,
               "grid": f"{side}x{side}", "pos_thresh": args.pos_thresh, "classes": {}}

        for change_class, query_text in [(c, q) for q, c in query_to_mask_class.items()]:
            t = encode_query(enc, query_text)
            hits, aps, rands, n_used = [], [], [], 0
            for i, pk in enumerate(pairs):
                if not ds.has_mask(pk.location_id):
                    continue
                mask = ds.load_change_mask(pk, change_class)
                if not mask.any():
                    continue
                gt = _gt_patches(mask, side, args.pos_thresh)
                if not gt.any():            # change too small to land on a patch
                    continue
                delta = P2[i] @ t - P1[i] @ t          # [n_patch] Δ-similarity
                hits.append(bool(gt[int(np.argmax(delta))]))
                aps.append(_average_precision(delta, gt))
                rands.append(float(gt.mean()))
                n_used += 1
            res = {
                "query": query_text,
                "n_pairs": n_used,
                "pointing_game": round(float(np.mean(hits)), 4) if hits else None,
                "random_pointing": round(float(np.mean(rands)), 4) if rands else None,
                "patch_AP": round(float(np.mean(aps)), 4) if aps else None,
                "prevalence_floor": round(float(np.mean(rands)), 4) if rands else None,
            }
            out["classes"][change_class] = res
            print(f"{enc_name:11} {change_class:8} N={n_used:4d}  "
                  f"pointing={res['pointing_game']} (rand {res['random_pointing']})  "
                  f"patch_AP={res['patch_AP']}")

        op = Path(args.results_dir) / f"localization_{args.dataset}__{enc_name}__{args.split}.json"
        op.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"  wrote {op}")


if __name__ == "__main__":
    main()
