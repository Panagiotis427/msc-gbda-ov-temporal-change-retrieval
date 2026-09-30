"""The per-patch tokens of ``OpenClipHFEncoder._patch_tokens`` must equal open_clip's own contextual token
stream, and must not depend on which other images share the batch. A random tiny ViT keeps it offline and fast."""
import pytest
import torch

open_clip = pytest.importorskip("open_clip")
from open_clip.transformer import VisionTransformer  # noqa: E402

from src.encoders._openclip_base import OpenClipHFEncoder  # noqa: E402


class _Stub:
    """Just enough of an encoder for the unbound method: ``_model.visual``."""
    def __init__(self, visual):
        self._model = type("M", (), {"visual": visual})()


@pytest.fixture(scope="module")
def vit():
    torch.manual_seed(0)
    return VisionTransformer(image_size=32, patch_size=16, width=16, layers=2, heads=2, mlp_ratio=2.0,
                             output_dim=8).eval()


def _reference(v, px):
    """open_clip's forward up to the projected tokens (CLS dropped): its own ``_embeds`` where the release has
    it (3.x), else the same steps in the transformer's own layout."""
    if hasattr(v, "_embeds"):
        return (v.ln_post(v.transformer(v._embeds(px))) @ v.proj)[:, 1:, :]
    x = v.conv1(px)
    x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
    cls = v.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype)
    x = v.ln_pre(torch.cat([cls, x], dim=1) + v.positional_embedding.to(x.dtype))
    if not getattr(v.transformer, "batch_first", False):
        x = v.transformer(x.permute(1, 0, 2)).permute(1, 0, 2)
    else:
        x = v.transformer(x)
    return (v.ln_post(x) @ v.proj)[:, 1:, :]


def test_patch_tokens_equal_open_clip_token_stream(vit):
    px = torch.randn(3, 3, 32, 32)
    with torch.no_grad():
        got = OpenClipHFEncoder._patch_tokens(_Stub(vit), px)
        ref = _reference(vit, px)
    assert got is not None and got.shape == ref.shape == (3, 4, 8)
    assert torch.allclose(got, ref, atol=1e-5)


def test_patch_tokens_do_not_depend_on_batch_mates(vit):
    px = torch.randn(3, 3, 32, 32)
    with torch.no_grad():
        batched = OpenClipHFEncoder._patch_tokens(_Stub(vit), px)
        single = torch.cat([OpenClipHFEncoder._patch_tokens(_Stub(vit), px[i:i + 1]) for i in range(3)], 0)
    assert torch.allclose(batched, single, atol=1e-5)


def test_patches_attend_to_each_other(vit):
    """Changing one patch of an image must change the tokens of the others (contextual encoding)."""
    px = torch.randn(1, 3, 32, 32)
    px2 = px.clone()
    px2[:, :, :16, :16] += 1.0                           # the first of the four 16x16 patches
    with torch.no_grad():
        a = OpenClipHFEncoder._patch_tokens(_Stub(vit), px)
        b = OpenClipHFEncoder._patch_tokens(_Stub(vit), px2)
    assert not torch.allclose(a[:, 1:, :], b[:, 1:, :], atol=1e-6)
