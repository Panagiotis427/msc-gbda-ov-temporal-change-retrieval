"""
Shared base for ``open_clip``-architecture encoders whose weights ship as a
plain checkpoint in a HuggingFace repo (GeoRSCLIP, RemoteCLIP).

Both follow the same recipe: build a standard open_clip CLIP of a given
architecture, then load a domain-specific state dict downloaded with
``huggingface_hub.hf_hub_download``. They expose the project's
``ImageTextEncoder`` protocol.
"""
from __future__ import annotations

import contextlib
import hashlib
import logging
from typing import Iterator, List, Optional, Union

from src import _cache  # noqa: F401  sets HF_HOME before huggingface_hub
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


@contextlib.contextmanager
def _silence_open_clip_random_init() -> Iterator[None]:
    """Suppress open_clip's spurious ``No pretrained weights loaded ... Model
    initialized randomly`` warning emitted while building the bare architecture.
    It is misleading here: we immediately load the domain checkpoint over the
    random init (see :func:`_load_state_dict_flexible`), so the model is never
    left random. Scoped to the model-creation call only."""

    class _Filter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
            return "No pretrained weights loaded" not in record.getMessage()

    root = logging.getLogger()
    flt = _Filter()
    root.addFilter(flt)
    try:
        yield
    finally:
        root.removeFilter(flt)


def _sha256(path: str) -> str:
    """Streaming SHA-256 of a file (checkpoints are large, so hash in chunks)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_state_dict_flexible(
    model, ckpt_path: str, expected_sha256: Optional[str] = None
) -> None:
    """Tolerant load: handles ``{'state_dict': ...}`` wrappers, ``module.``
    prefixes, and partial matches (strict=False).

    Supply-chain guard: when *expected_sha256* is set (see
    ``OpenClipHFEncoder._hf_sha256``), the downloaded checkpoint's digest is
    verified *before* it is ever handed to ``torch.load``, and a mismatch refuses
    the load — closing the "compromised upstream repo -> arbitrary unpickle ->
    code execution" path for the ``weights_only=False`` fallback below.
    """
    if expected_sha256:
        actual = _sha256(ckpt_path)
        if actual != expected_sha256:
            raise RuntimeError(
                f"{ckpt_path}: checksum mismatch (expected {expected_sha256}, got "
                f"{actual}); refusing to load an unverified checkpoint."
            )
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001 — fall back loudly, never silently
        # These domain checkpoints (GeoRSCLIP/RemoteCLIP) bundle non-tensor objects,
        # so the safe loader legitimately fails and the unsafe path is the normal
        # one. Guard it: with a checksum configured the bytes are already verified;
        # without one, warn that pinning ``_hf_sha256`` (+ ``_hf_revision``) would
        # harden this (never fall back silently).
        verified = "checksum verified" if expected_sha256 else (
            "no checksum configured — pin _hf_sha256 + _hf_revision to harden")
        print(f"  [security] {ckpt_path}: weights_only load failed ({exc}); "
              f"retrying with weights_only=False ({verified}).")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # CLIP logit_scale / position-id buffers are commonly absent — only worry
    # if entire towers failed to load: the whole vision tower, and the text tower's
    # transformer plus its embeddings, final norm and projection (the key names of
    # open_clip's CLIP, which both encoders are built on).
    critical = [m for m in missing if m.startswith(
        ("visual.", "transformer.", "token_embedding", "positional_embedding",
         "ln_final", "text_projection"))]
    if critical:
        raise RuntimeError(
            f"Checkpoint missing critical weights (e.g. {critical[:3]} ...). "
            "Architecture/checkpoint mismatch."
        )


class OpenClipHFEncoder:
    """Concrete encoders set ``name``, ``embed_dim``, ``_arch``,
    ``_hf_repo``, ``_hf_file``."""

    name: str = "openclip"
    embed_dim: int = 512
    # 2 = patch tokens in the transformer's own layout (a07b8a6); caches of per-patch embeddings written
    # before carry no marker and are not reused for these encoders (see patch_cache_path)
    patch_tokens_version: int = 2
    image_input_size: int = 224
    _arch: str = "ViT-B-32"
    _hf_repo: str = ""
    _hf_file: str = ""
    # Supply-chain pinning (recommended for the public deployment): set
    # ``_hf_revision`` to a commit SHA so the download can't silently change under
    # us, and ``_hf_sha256`` to the checkpoint's digest so a swapped file is
    # rejected before the ``weights_only=False`` load. Both default to None
    # (fetch latest / no integrity check) to preserve current behaviour.
    _hf_revision: Optional[str] = None
    _hf_sha256: Optional[str] = None

    def __init__(
        self,
        device: Optional[torch.device] = None,
        cache_dir: Optional[str] = None,
    ) -> None:
        try:
            import open_clip
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                f"open-clip-torch is required for {self.name}. "
                "Install: pip install open-clip-torch"
            ) from exc
        from huggingface_hub import hf_hub_download

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(f"Loading {self.name}: {self._hf_repo}/{self._hf_file} "
              f"(arch {self._arch})")
        with _silence_open_clip_random_init():
            model, _, preprocess = open_clip.create_model_and_transforms(self._arch)
        ckpt_path = hf_hub_download(
            repo_id=self._hf_repo, filename=self._hf_file,
            revision=self._hf_revision, cache_dir=cache_dir,
        )
        _load_state_dict_flexible(model, ckpt_path, expected_sha256=self._hf_sha256)

        self._model = model.to(self.device).eval()
        for p in self._model.parameters():
            p.requires_grad = False
        self._preprocess = preprocess
        self._tokenizer = open_clip.get_tokenizer(self._arch)

    # ------------------------------------------------------------------
    def encode_text(
        self, texts: Union[str, List[str]], batch_size: int = 32
    ) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        out: List[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                tok = self._tokenizer(texts[i:i + batch_size]).to(self.device)
                f = F.normalize(self._model.encode_text(tok), dim=-1)
                out.append(f.cpu().numpy().astype(np.float32))
        return np.concatenate(out, axis=0)

    def encode_image(
        self, images: Union[Image.Image, List[Image.Image]], batch_size: int = 32
    ) -> np.ndarray:
        if isinstance(images, Image.Image):
            images = [images]
        out: List[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                px = torch.stack(
                    [self._preprocess(im) for im in images[i:i + batch_size]]
                ).to(self.device)
                f = F.normalize(self._model.encode_image(px), dim=-1)
                out.append(f.cpu().numpy().astype(np.float32))
        return np.concatenate(out, axis=0)

    # ------------------------------------------------------------------
    def _patch_tokens(self, px: torch.Tensor) -> Optional[torch.Tensor]:
        """Replicate open_clip VisionTransformer forward, returning per-patch
        features projected into the shared space ``[1, N, D]``. Returns None
        if the visual tower is not a standard open_clip ViT."""
        v = self._model.visual
        try:
            x = v.conv1(px)                                  # [1, w, gh, gw]
            x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
            cls = v.class_embedding.to(x.dtype) + torch.zeros(
                x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
            x = torch.cat([cls, x], dim=1)
            x = x + v.positional_embedding.to(x.dtype)
            x = v.ln_pre(x)
            # open_clip >= 2.24 builds the transformer batch-first ([N, L, D]); earlier releases are
            # sequence-first ([L, N, D]). Handing a batch-first transformer the sequence-first layout
            # makes every patch attend only to itself (one image per call) or to the same position in
            # the other images of a batch, so the layout follows the transformer.
            seq_first = not getattr(v.transformer, "batch_first", False)
            if seq_first:
                x = x.permute(1, 0, 2)
            x = v.transformer(x)
            if seq_first:
                x = x.permute(1, 0, 2)
            x = v.ln_post(x)
            if getattr(v, "proj", None) is not None:
                x = x @ v.proj
            return x[:, 1:, :]                                # drop CLS
        except Exception:
            return None

    def encode_image_patches(
        self,
        image: Union[Image.Image, List[Image.Image]],
        batch_size: int = 32,
    ) -> Optional[np.ndarray]:
        """Per-patch embeddings projected into the shared space, L2-normalised,
        as a ``[n_patches, D]`` float32 array (raw cosine-comparable — no per-image
        min-max, unlike ``compute_patch_text_similarity``). ``None`` if the visual
        tower is not a standard open_clip ViT. Used by patch-level retrieval.

        Accepts a single image (returns ``[n_patches, D]``) or a list (returns
        ``[N, n_patches, D]``), encoding lists in GPU batches of ``batch_size``."""
        single = isinstance(image, Image.Image)
        images = [image] if single else list(image)
        out: List[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                px = torch.stack(
                    [self._preprocess(im) for im in images[i:i + batch_size]]
                ).to(self.device)
                patches = self._patch_tokens(px)               # [B, N, D] or None
                if patches is None:
                    return None
                pf = F.normalize(patches, dim=-1)
                out.append(pf.cpu().numpy().astype(np.float32))
        arr = np.concatenate(out, axis=0)                      # [len(images), N, D]
        return arr[0] if single else arr

    # ------------------------------------------------------------------
    def lora_visual_spec(self):
        """LoRA seam for open_clip-architecture visual towers (see
        :class:`src.encoders.base.LoRAVisualSpec`). The visual tower is a single
        callable returning projected image features, so ``forward`` is a one-liner;
        LoRA targets the ResBlock FFN (``c_fc``/``c_proj``) — adapting attention
        would be a silent no-op (see :mod:`src.lora_train`)."""
        from .base import LoRAVisualSpec

        def _set(module) -> None:
            self._model.visual = module

        def _forward(module, px: torch.Tensor) -> torch.Tensor:
            return F.normalize(module(px), dim=-1)

        return LoRAVisualSpec(
            module=self._model.visual,
            target_modules=["c_fc", "c_proj"],
            preprocess=self._preprocess,          # torchvision transform: PIL -> [C,H,W]
            forward=_forward,
            set_module=_set,
            to_device=lambda dev: self._model.to(dev),
        )

    def compute_patch_text_similarity(
        self, image: Image.Image, text: str
    ) -> np.ndarray:
        with torch.no_grad():
            px = self._preprocess(image).unsqueeze(0).to(self.device)
            patches = self._patch_tokens(px)
            tok = self._tokenizer([text]).to(self.device)
            tfeat = F.normalize(self._model.encode_text(tok), dim=-1)  # [1, D]
            if patches is None:
                f = F.normalize(self._model.encode_image(px), dim=-1)
                return np.array([[float((f @ tfeat.t()).item())]], np.float32)
            pf = F.normalize(patches, dim=-1)                          # [1,N,D]
            sims = (pf @ tfeat.t()).squeeze(0).squeeze(-1)             # [N]
            n = sims.shape[0]
            side = int(round(n ** 0.5))
            if side * side != n:
                return np.array([[float(sims.mean().item())]], np.float32)
            grid = sims.view(side, side).cpu().numpy().astype(np.float32)
        lo, hi = float(grid.min()), float(grid.max())
        if hi - lo < 1e-8:
            return np.zeros_like(grid, dtype=np.float32)
        return ((grid - lo) / (hi - lo)).astype(np.float32)
