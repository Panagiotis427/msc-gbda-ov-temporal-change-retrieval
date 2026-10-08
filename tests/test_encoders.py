"""Protocol-conformance tests for encoder implementations (mock-based, no CLIP download)."""
import numpy as np
import pytest
import torch
from PIL import Image

from src.encoders.base import ImageTextEncoder


class _MockEncoder:
    """Minimal conforming implementation of ImageTextEncoder for protocol tests."""
    name = "mock"
    embed_dim = 64
    image_input_size = 224

    def __init__(self):
        self.device = torch.device("cpu")

    def encode_text(self, texts, batch_size=32):
        if isinstance(texts, str):
            texts = [texts]
        n = len(texts)
        out = np.random.randn(n, self.embed_dim).astype(np.float32)
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return out / norms

    def encode_image(self, images, batch_size=32):
        if isinstance(images, Image.Image):
            images = [images]
        n = len(images)
        out = np.random.randn(n, self.embed_dim).astype(np.float32)
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return out / norms

    def compute_patch_text_similarity(self, image, text):
        grid = np.random.rand(14, 14).astype(np.float32)
        return (grid - grid.min()) / (grid.max() - grid.min() + 1e-8)


class TestImageTextEncoderProtocol:
    """Verify _MockEncoder satisfies the protocol at runtime."""

    def test_isinstance_check(self):
        enc = _MockEncoder()
        assert isinstance(enc, ImageTextEncoder)

    def test_required_attributes(self):
        enc = _MockEncoder()
        assert isinstance(enc.name, str)
        assert isinstance(enc.embed_dim, int)
        assert isinstance(enc.image_input_size, int)
        assert isinstance(enc.device, torch.device)

    def test_encode_text_shape(self):
        enc = _MockEncoder()
        result = enc.encode_text("test query")
        assert result.shape == (1, enc.embed_dim)
        assert result.dtype == np.float32

    def test_encode_text_batch(self):
        enc = _MockEncoder()
        texts = ["query one", "query two", "query three"]
        result = enc.encode_text(texts)
        assert result.shape == (3, enc.embed_dim)

    def test_encode_text_l2_normalised(self):
        enc = _MockEncoder()
        result = enc.encode_text(["hello world"])
        norms = np.linalg.norm(result, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)

    def test_encode_image_shape(self):
        enc = _MockEncoder()
        img = Image.new("RGB", (224, 224))
        result = enc.encode_image(img)
        assert result.shape == (1, enc.embed_dim)

    def test_encode_image_batch(self):
        enc = _MockEncoder()
        imgs = [Image.new("RGB", (224, 224)) for _ in range(4)]
        result = enc.encode_image(imgs)
        assert result.shape == (4, enc.embed_dim)

    def test_patch_similarity_shape_and_range(self):
        enc = _MockEncoder()
        img = Image.new("RGB", (224, 224))
        grid = enc.compute_patch_text_similarity(img, "forest")
        assert grid.ndim == 2
        assert grid.dtype == np.float32
        assert float(grid.min()) >= 0.0 - 1e-6
        assert float(grid.max()) <= 1.0 + 1e-6


class TestEncoderRegistry:
    def test_get_encoder_unknown_raises(self):
        from src.encoders import get_encoder
        with pytest.raises(ValueError, match="Unknown encoder"):
            get_encoder("not_a_real_encoder")

    def test_registered_encoders_listed(self):
        from src.encoders import _FACTORIES
        assert "clip_vitl14" in _FACTORIES
        assert "georsclip" in _FACTORIES
        assert "remoteclip" in _FACTORIES


class TestConcreteEncoderClassContracts:
    """Structural checks on the real encoder classes — no weights downloaded."""

    def test_rs_encoder_class_attributes(self):
        from src.encoders.georsclip import GeoRSCLIPEncoder
        from src.encoders.remoteclip import RemoteCLIPEncoder

        for cls, dim, arch in (
            (GeoRSCLIPEncoder, 512, "ViT-B-32"),
            (RemoteCLIPEncoder, 768, "ViT-L-14"),
        ):
            assert cls.embed_dim == dim
            assert cls.image_input_size == 224
            assert cls._arch == arch
            assert cls._hf_repo and cls._hf_file
            for m in ("encode_text", "encode_image",
                      "compute_patch_text_similarity"):
                assert callable(getattr(cls, m))

    def test_clip_encoder_class_attributes(self):
        from src.encoders.clip_vitl14 import CLIPViTL14Encoder
        assert CLIPViTL14Encoder.embed_dim == 768
        assert CLIPViTL14Encoder.name == "clip_vitl14"

    def test_clip_image_processor_kwargs_follow_transformers(self):
        # transformers 5.4 replaced `use_fast` with `backend`; both pick the torchvision processor
        from src.encoders.clip_vitl14 import _image_processor_kwargs
        assert _image_processor_kwargs("4.36.2") == {"use_fast": True}
        assert _image_processor_kwargs("5.3.0") == {"use_fast": True}
        assert _image_processor_kwargs("5.4.0") == {"backend": "torchvision"}
        assert _image_processor_kwargs("5.19.0") == {"backend": "torchvision"}


class TestCheckpointLoadGuard:
    """``_load_state_dict_flexible`` refuses a checkpoint that lacks the text tower's
    embeddings, final norm or projection, not only one that lacks a whole tower
    (tiny CPU module with open_clip's key names; no weights)."""

    @staticmethod
    def _tiny():
        m = torch.nn.Module()
        m.token_embedding = torch.nn.Embedding(4, 2)
        m.ln_final = torch.nn.LayerNorm(2)
        m.text_projection = torch.nn.Parameter(torch.zeros(2, 2))
        m.positional_embedding = torch.nn.Parameter(torch.zeros(3, 2))
        m.logit_scale = torch.nn.Parameter(torch.zeros(()))
        return m

    def _checkpoint(self, tmp_path, drop):
        sd = {k: v for k, v in self._tiny().state_dict().items() if not k.startswith(drop)}
        path = tmp_path / "ckpt.pt"
        torch.save(sd, path)
        return str(path)

    @pytest.mark.parametrize("name", ["token_embedding", "positional_embedding",
                                      "ln_final", "text_projection"])
    def test_missing_text_tower_weights_are_refused(self, tmp_path, name):
        from src.encoders._openclip_base import _load_state_dict_flexible
        with pytest.raises(RuntimeError, match="missing critical weights"):
            _load_state_dict_flexible(self._tiny(), self._checkpoint(tmp_path, (name,)))

    def test_missing_logit_scale_is_tolerated(self, tmp_path):
        from src.encoders._openclip_base import _load_state_dict_flexible
        _load_state_dict_flexible(self._tiny(), self._checkpoint(tmp_path, ("logit_scale",)))
