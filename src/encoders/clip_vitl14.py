"""
OpenAI CLIP ViT-L/14 wrapped as an `ImageTextEncoder`.

Composes the existing `FrozenTextEncoder` for text and loads a `CLIPModel` /
`CLIPProcessor` pair for the image side, exposing `_clip_model` and `_processor`.
"""
from __future__ import annotations

from src import _cache  # noqa: F401  sets HF_HOME before transformers
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from ..text_encoder import FrozenTextEncoder


_DEFAULT_MODEL = "openai/clip-vit-large-patch14"


class CLIPViTL14Encoder:
    """`ImageTextEncoder` for OpenAI CLIP ViT-L/14 (768-d shared space)."""

    name = "clip_vitl14"
    embed_dim = 768     # ViT-L/14's width; __init__ sets the instance value from the loaded model
    image_input_size = 224

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        device: Optional[torch.device] = None,
        cache_dir: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_dir = cache_dir or _cache.CLIP_CACHE_DIR

        self._text = FrozenTextEncoder(
            model_name=model_name,
            device=self.device,
            cache_dir=cache_dir,
        )

        print(f"Loading CLIP vision tower: {model_name}")
        self._clip_model: CLIPModel = CLIPModel.from_pretrained(model_name, cache_dir=self.cache_dir).to(self.device)
        # The class attribute is the ViT-L/14 value; follow the loaded model when
        # `model_name` points at another CLIP (the name and cache keys stay "clip_vitl14").
        self.embed_dim = int(self._clip_model.visual_projection.out_features)
        self._clip_model.eval()
        for p in self._clip_model.parameters():
            p.requires_grad = False
        self._processor: CLIPProcessor = CLIPProcessor.from_pretrained(
            model_name, cache_dir=self.cache_dir, use_fast=True
        )

    def encode_text(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
    ) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        embs: List[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            t = self._text.encode(batch)
            t = F.normalize(t, dim=-1)
            embs.append(t.detach().cpu().numpy())
        return np.concatenate(embs, axis=0)

    def encode_image(
        self,
        images: Union[Image.Image, List[Image.Image]],
        batch_size: int = 32,
    ) -> np.ndarray:
        if isinstance(images, Image.Image):
            images = [images]
        embs: List[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch = images[i : i + batch_size]
                pixel_values = self._processor(images=batch, return_tensors="pt").pixel_values.to(self.device)
                vision_out = self._clip_model.vision_model(pixel_values=pixel_values)
                feats = self._clip_model.visual_projection(vision_out.pooler_output)
                feats = F.normalize(feats, dim=-1)
                embs.append(feats.cpu().numpy())
        return np.concatenate(embs, axis=0)

    def encode_image_patches(
        self,
        image: Union[Image.Image, List[Image.Image]],
        batch_size: int = 32,
    ) -> np.ndarray:
        """Per-patch embeddings projected into the shared CLIP space, L2-normalised,
        as a ``[n_patches, D]`` float32 array (raw cosine-comparable — no per-image
        min-max). ViT-L/14 @224 → 256 patches. Used by patch-level retrieval.

        Accepts a single image (returns ``[n_patches, D]``) or a list (returns
        ``[N, n_patches, D]``), encoding lists in GPU batches of ``batch_size``."""
        single = isinstance(image, Image.Image)
        images = [image] if single else list(image)
        out: List[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch = images[i:i + batch_size]
                pixel_values = self._processor(images=batch, return_tensors="pt").pixel_values.to(self.device)
                vision_outputs = self._clip_model.vision_model(pixel_values=pixel_values)
                # Deliberately read the raw patch tokens (last_hidden_state) without
                # the vision tower's post_layernorm — HF CLIP applies that LN only to
                # the pooled CLS token, so dense per-patch features conventionally
                # skip it (the MaskCLIP-style recipe). NOTE: the open_clip encoders
                # (_openclip_base._patch_tokens) DO apply ln_post to patches, so the
                # two encoder families normalise patches slightly differently. This
                # is harmless for patch-level retrieval because every comparison is
                # within a single encoder (patch_eval never mixes encoders); it would
                # only matter if patch embeddings were ever compared across families.
                patch_tokens = vision_outputs.last_hidden_state[:, 1:, :]
                projected = self._clip_model.visual_projection(patch_tokens)
                projected = F.normalize(projected, dim=-1)             # [B, N, D]
                out.append(projected.cpu().numpy().astype(np.float32))
        arr = np.concatenate(out, axis=0)                              # [len(images), N, D]
        return arr[0] if single else arr

    def lora_visual_spec(self):
        """LoRA seam for the HF-transformers CLIP vision tower (see
        :class:`src.encoders.base.LoRAVisualSpec`).

        The trainable module is ``vision_model``; producing a shared-space
        embedding is a two-step (``vision_model`` pooled output → ``visual_projection``),
        wrapped here so the trainer sees the same ``forward`` contract as the
        open_clip family. LoRA targets the HF CLIP encoder-layer MLP
        (``fc1``/``fc2``) — the transformers analogue of open_clip's ``c_fc``/``c_proj``.
        """
        from .base import LoRAVisualSpec

        def _set(module) -> None:
            self._clip_model.vision_model = module

        def _preprocess(image: Image.Image) -> torch.Tensor:
            return self._processor(images=image, return_tensors="pt").pixel_values[0]

        def _forward(module, px: torch.Tensor) -> torch.Tensor:
            vision_out = module(pixel_values=px)
            feats = self._clip_model.visual_projection(vision_out.pooler_output)
            return F.normalize(feats, dim=-1)

        return LoRAVisualSpec(
            module=self._clip_model.vision_model,
            target_modules=["fc1", "fc2"],
            preprocess=_preprocess,
            forward=_forward,
            set_module=_set,
            to_device=lambda dev: self._clip_model.to(dev),
        )

    def compute_patch_text_similarity(
        self,
        image: Image.Image,
        text: str,
    ) -> np.ndarray:
        """Per-patch cosine similarity, projected into the shared CLIP space.

        Returns a ``[grid_h, grid_w]`` float32 array in ``[0, 1]`` derived from
        per-patch · text-embedding cosine similarity. For ViT-L/14 at 224 input
        the grid is 16×16.
        """
        with torch.no_grad():
            pixel_values = self._processor(images=image, return_tensors="pt").pixel_values.to(self.device)
            vision_outputs = self._clip_model.vision_model(pixel_values=pixel_values)
            patch_tokens = vision_outputs.last_hidden_state[:, 1:, :]
            projected = self._clip_model.visual_projection(patch_tokens)
            projected = F.normalize(projected, dim=-1)

            text_inputs = self._processor.tokenizer(
                [text],
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt",
            ).to(self.device)
            text_out = self._clip_model.text_model(**text_inputs)
            text_features = self._clip_model.text_projection(text_out.pooler_output)
            text_features = F.normalize(text_features, dim=-1)

            sims = (projected @ text_features.t()).squeeze(0).squeeze(-1)
            n_patches = sims.shape[0]
            grid_side = int(round(n_patches**0.5))
            if grid_side * grid_side != n_patches:
                raise RuntimeError(
                    f"Expected square patch grid; got {n_patches} patches (sqrt={grid_side})."
                )
            grid = sims.view(grid_side, grid_side).cpu().numpy().astype(np.float32)

        lo, hi = float(grid.min()), float(grid.max())
        if hi - lo < 1e-8:
            return np.zeros_like(grid, dtype=np.float32)
        return ((grid - lo) / (hi - lo)).astype(np.float32)
