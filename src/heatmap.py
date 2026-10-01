"""
Spatial Heatmap Generation for Change Localization.

Public API:
    generate_heatmap(image_t1, image_t2, text, encoder, alpha=0.5)
        — query-vs-After-image similarity (T1 ignored).
    generate_change_heatmap(image_t1, image_t2, text, encoder, alpha=0.5)
        — query-conditioned CHANGE heatmap: per-patch Δ-similarity T1->T2.
    extract_patch_attention(image_t2, image_t1, encoder) -> np.ndarray [grid_h, grid_w]
    extract_attention_weights(image, text_query, encoder) -> np.ndarray [grid_h, grid_w]
    resize_heatmap / apply_overlay / generate_grid_heatmap_from_patches  (unchanged)

All vision logic is delegated to ``encoder.compute_patch_text_similarity()``, which must
return a float32 [grid_h, grid_w] array normalised to [0, 1].
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

if TYPE_CHECKING:
    from .encoders.base import ImageTextEncoder


def generate_heatmap(
    image_t1: np.ndarray,
    image_t2: np.ndarray,
    text: str,
    encoder: "ImageTextEncoder",
    alpha: float = 0.5,
) -> Tuple[np.ndarray, Image.Image]:
    """Return (raw_heatmap_grid, blended_PIL_image) for image_t2 against *text*.

    Args:
        image_t1: H×W×3 uint8 ndarray (used only for patch-diff path — ignored here).
        image_t2: H×W×3 uint8 ndarray displayed underneath the heatmap.
        text: Natural-language change query.
        encoder: Any ``ImageTextEncoder`` implementation.
        alpha: Heatmap opacity (0 = transparent, 1 = opaque).

    Returns:
        tuple: (heatmap_grid [grid_h, grid_w] float32 in [0,1],
                blended PIL.Image.Image)
    """
    pil_t2 = Image.fromarray(image_t2) if isinstance(image_t2, np.ndarray) else image_t2
    heatmap = encoder.compute_patch_text_similarity(pil_t2, text)   # [grid_h, grid_w]
    h, w = image_t2.shape[:2] if isinstance(image_t2, np.ndarray) else (image_t2.height, image_t2.width)
    heatmap_resized = resize_heatmap(heatmap, h, w)
    img_arr = np.array(image_t2) if not isinstance(image_t2, np.ndarray) else image_t2
    blended = apply_overlay(img_arr, heatmap_resized, alpha=alpha)
    return heatmap, blended


def generate_change_heatmap(
    image_t1: np.ndarray,
    image_t2: np.ndarray,
    text: str,
    encoder: "ImageTextEncoder",
    alpha: float = 0.5,
    *,
    precomputed_p1: Optional[np.ndarray] = None,
    precomputed_p2: Optional[np.ndarray] = None,
) -> Tuple[Optional[np.ndarray], Optional[Image.Image]]:
    """Query-conditioned **change** heatmap (the honest localiser for this engine).

    Localises where the query's presence *grew* from T1 to T2 via the per-patch
    Δ-similarity ``cos(t, P2_p) - cos(t, P1_p)`` (the same signal as the S3
    patch-level scorer, report §8.1) — unlike ``generate_heatmap`` which
    only matches the query against the After image and ignores T1.

    Pass ``precomputed_p1``/``precomputed_p2`` (``[n_patch, D]`` L2-normed patch
    embeddings, as ``encode_image_patches`` returns) to skip re-encoding when the
    caller already has them cached (the ``patch`` approach caches the whole corpus).

    Returns ``(grid [0,1], blended PIL)`` or ``(None, None)`` if the encoder
    exposes no patch tokens (caller should fall back to ``generate_heatmap``).
    """
    if precomputed_p1 is not None and precomputed_p2 is not None:
        P1, P2 = precomputed_p1, precomputed_p2
    else:
        patches_fn = getattr(encoder, "encode_image_patches", None)
        if patches_fn is None:
            return None, None
        pil_t1, pil_t2 = _to_pil(image_t1), _to_pil(image_t2)
        P1, P2 = patches_fn(pil_t1), patches_fn(pil_t2)
    if P1 is None or P2 is None:
        return None, None

    from src.benchmark import encode_query
    t = encode_query(encoder, text)               # L2-normed query vector
    delta = (P2 @ t) - (P1 @ t)                    # [n_patch]
    n = int(delta.shape[0])
    side = int(round(n ** 0.5))
    grid = delta.reshape(side, side) if side * side == n else delta.reshape(1, n)
    lo, hi = float(grid.min()), float(grid.max())
    norm = (np.zeros_like(grid) if hi - lo < 1e-8
            else (grid - lo) / (hi - lo)).astype(np.float32)

    h, w = (image_t2.shape[:2] if isinstance(image_t2, np.ndarray)
            else (image_t2.height, image_t2.width))
    resized = resize_heatmap(norm, h, w)
    img_arr = np.array(image_t2) if not isinstance(image_t2, np.ndarray) else image_t2
    return norm, apply_overlay(img_arr, resized, alpha=alpha)


def extract_patch_attention(
    image_t2,
    image_t1,
    encoder: Optional["ImageTextEncoder"] = None,
    model=None,
) -> np.ndarray:
    """Patch-difference heatmap between T2 and T1.

    Supports the new encoder protocol (``encoder`` kwarg) **and** the legacy
    ``model`` kwarg used by existing tests so they keep passing.

    Returns float32 [grid_h, grid_w] in [0, 1].
    """
    # --- legacy model path (tests use MockCLIPModel with vision_encoder attr) ---
    if encoder is None:
        return _legacy_patch_diff(image_t2, image_t1, model)

    # --- encoder protocol path ---
    pil_t1 = _to_pil(image_t1)
    pil_t2 = _to_pil(image_t2)
    sim_t1 = encoder.compute_patch_text_similarity(pil_t1, "")
    sim_t2 = encoder.compute_patch_text_similarity(pil_t2, "")
    diff = np.abs(sim_t2.astype(np.float32) - sim_t1.astype(np.float32))
    lo, hi = diff.min(), diff.max()
    if hi - lo < 1e-8:
        return np.zeros_like(diff, dtype=np.float32)
    return ((diff - lo) / (hi - lo)).astype(np.float32)


def extract_attention_weights(
    image: torch.Tensor,
    text_query: str,
    encoder: Optional["ImageTextEncoder"] = None,
    model=None,
    target_patch: Optional[int] = None,
) -> np.ndarray:
    """Patch–text similarity heatmap for *image* given *text_query*.

    Supports the new encoder protocol (``encoder`` kwarg) **and** the legacy
    ``model`` kwarg used by existing tests.

    Returns float32 [grid_h, grid_w] in [0, 1].
    """
    # --- encoder protocol path ---
    if encoder is not None:
        pil = _to_pil(image)
        return encoder.compute_patch_text_similarity(pil, text_query)

    # --- legacy model path (MockCLIPModel in tests) ---
    return _legacy_attention_weights(image, text_query, model, target_patch)


# ---------------------------------------------------------------------------
# Utility functions (unchanged API, keep tests green)
# ---------------------------------------------------------------------------

def resize_heatmap(
    heatmap: np.ndarray,
    target_height: int,
    target_width: int,
) -> np.ndarray:
    """Bilinear-resize a patch grid to full image resolution.

    Resizing is done in ``float32`` (no ``uint8`` round-trip), so no precision is
    lost. The output is clipped to ``[0, 1]``, which means callers must pass a
    heatmap already normalised to ``[0, 1]`` — a **signed / raw-cosine Δ map fed
    in directly loses its negative half to the clip**. For change scoring on patch
    features (where the absolute, T1/T2-comparable cosine matters), use
    ``encode_image_patches`` and difference the raw per-patch cosines instead of
    routing them through this display resizer.
    """
    resized = cv2.resize(
        heatmap.astype(np.float32),
        (target_width, target_height),
        interpolation=cv2.INTER_LINEAR,
    )
    return np.clip(resized, 0, 1)


def apply_overlay(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: str = "jet",
) -> Image.Image:
    """Alpha-blend colorised heatmap over image."""
    heat_colormap = cv2.applyColorMap(
        (np.clip(heatmap, 0, 1) * 255).astype(np.uint8),
        getattr(cv2, f"COLORMAP_{colormap.upper()}", cv2.COLORMAP_JET),
    )
    heat_rgb = cv2.cvtColor(heat_colormap, cv2.COLOR_BGR2RGB)
    blended = cv2.addWeighted(image.astype(np.uint8), 1 - alpha, heat_rgb, alpha, 0)
    return Image.fromarray(blended)


def generate_grid_heatmap_from_patches(
    patch_scores: np.ndarray,
    grid_shape: Tuple[int, int] = (12, 12),
    out_size: Optional[Tuple[int, int]] = None,
    colormap: str = "jet",
) -> Tuple[np.ndarray, np.ndarray]:
    """Reshape flat patch scores to a ``[grid_h, grid_w]`` grid and a colormap image.

    By default the colormap is rendered at the grid resolution. Pass
    ``out_size=(height, width)`` to bicubic-upsample the grid to a full image
    resolution first (the earlier version always resized the grid to its own shape
    — a no-op despite the "full-res" name). Returns ``(patch_grid, colormap_bgr)``.
    """
    patch_grid = (
        patch_scores.reshape(grid_shape).astype(np.float32)
        if patch_scores.ndim == 1
        else patch_scores.astype(np.float32)
    )
    target_h, target_w = out_size if out_size is not None else patch_grid.shape[:2]
    heatmap_full = cv2.resize(
        patch_grid,
        (target_w, target_h),
        interpolation=cv2.INTER_CUBIC,
    )
    heatmap_colormap = cv2.applyColorMap(
        (np.clip(heatmap_full, 0, 1) * 255).astype(np.uint8),
        getattr(cv2, f"COLORMAP_{colormap.upper()}", cv2.COLORMAP_JET),
    )
    return patch_grid, heatmap_colormap


# ---------------------------------------------------------------------------
# Legacy helpers (keep tests passing without MockCLIPModel changes)
# ---------------------------------------------------------------------------

def _to_pil(img) -> Image.Image:
    if isinstance(img, Image.Image):
        return img
    if isinstance(img, np.ndarray):
        return Image.fromarray(img)
    # torch.Tensor [1, C, H, W] or [C, H, W]
    t = img
    if t.dim() == 4:
        t = t.squeeze(0)
    arr = (t.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def _get_legacy_vision_model(model):
    """Return a callable that maps image tensor → patch features."""
    if model is None:
        from transformers import AutoModel
        raw = AutoModel.from_pretrained("openai/clip-vit-large-patch14")
        return raw.vision_model
    if hasattr(model, "vision_encoder"):
        ve = model.vision_encoder
        return ve.forward if hasattr(ve, "forward") else ve
    if hasattr(model, "vision_model"):
        return model.vision_model
    raise ValueError("Model has no vision_model / vision_encoder attribute")


def _get_legacy_text_model(model):
    if model is None:
        return None
    if hasattr(model, "text_encoder"):
        te = model.text_encoder
        return te.forward if hasattr(te, "forward") else te
    if hasattr(model, "text_model"):
        return model.text_model
    return None


def _legacy_attention_weights(image, text_query, model, target_patch):
    vision_fn = _get_legacy_vision_model(model)
    text_fn = _get_legacy_text_model(model)

    with torch.no_grad():
        img_feats = vision_fn(image)          # [B, P, C]
        if target_patch is not None:
            img_feats = img_feats[:, target_patch:target_patch + 1, :]

        if text_fn is not None:
            text_feats = text_fn(text_query)  # [1, T, C]
        else:
            text_feats = torch.zeros(1, 1, img_feats.shape[-1])

        img_norm = img_feats / (img_feats.norm(dim=-1, keepdim=True) + 1e-8)
        txt_norm = text_feats / (text_feats.norm(dim=-1, keepdim=True) + 1e-8)
        sims = torch.matmul(img_norm.squeeze(0), txt_norm.squeeze(0).t())  # [P, T]

    if target_patch is not None:
        return np.array([[[float(sims.sum())]]])

    patch_scores = sims.sum(dim=-1).cpu().numpy()   # [P]
    n = patch_scores.shape[0]
    side = int(round(n ** 0.5))
    if side * side != n:
        side = int(n ** 0.5)
    lo, hi = patch_scores.min(), patch_scores.max()
    if hi - lo < 1e-8:
        return np.zeros((side, side), dtype=np.float32)
    normed = ((patch_scores - lo) / (hi - lo)).astype(np.float32)
    return np.clip(normed.reshape(side, side), 0, 1)


def _legacy_patch_diff(image_t2, image_t1, model):
    vision_fn = _get_legacy_vision_model(model)

    with torch.no_grad():
        f2 = vision_fn(image_t2)   # [B, P, C]
        f1 = vision_fn(image_t1)

        n2 = f2 / (f2.norm(dim=-1, keepdim=True) + 1e-8)
        n1 = f1 / (f1.norm(dim=-1, keepdim=True) + 1e-8)
        diff = torch.abs(n2 - n1).cpu().numpy()   # [B, P, C]

    patch_scores = diff[0].sum(axis=-1)   # [P]
    n = patch_scores.shape[0]
    side = int(round(n ** 0.5))
    if side * side != n:
        side = int(n ** 0.5)
    lo, hi = patch_scores.min(), patch_scores.max()
    if hi - lo < 1e-8:
        return np.zeros((side, side), dtype=np.float32)
    normed = ((patch_scores - lo) / (hi - lo)).astype(np.float32)
    return np.clip(normed.reshape(side, side), 0, 1)
