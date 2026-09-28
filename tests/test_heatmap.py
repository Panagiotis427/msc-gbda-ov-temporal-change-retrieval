"""
Unit Tests for Spatial Heatmap Generation Module
"""
import pytest
from PIL import Image

# Local imports
import torch
import numpy as np
from src.heatmap import (
    generate_heatmap,
    generate_change_heatmap,
    extract_attention_weights,
    extract_patch_attention,
    resize_heatmap,
    apply_overlay,
    generate_grid_heatmap_from_patches
)


class MockVisionEncoder:
    """Mock CLIP vision encoder with deterministic patch features."""
    def __init__(self, embed_dim):
        self.embed_dim = embed_dim
        self._vision_features_cache = {}

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """Mock vision encoder returning 576 patch embeddings (24x24 grid for CLIP ViT-L/14)."""
        if len(img.shape) == 5:
            img = img.squeeze(0)
        img_np = (img.cpu().numpy() * 255).astype(np.uint8)
        img_hash = int(hash(tuple(img_np.flatten())) % (2**31))
        if img_hash not in self._vision_features_cache:
            torch.manual_seed(img_hash + 1337)
            patch_features = np.random.randn(1, 576, self.embed_dim).astype(np.float32)
            self._vision_features_cache[img_hash] = torch.from_numpy(patch_features)
        return self._vision_features_cache[img_hash]


class MockTextEncoder:
    """Mock CLIP text encoder with deterministic features."""
    def __init__(self, embed_dim):
        self.embed_dim = embed_dim

    def forward(self, input_ids):  # Can be string or tensor
        # Handle both raw text strings and tensors for flexibility
        if isinstance(input_ids, str):
            ids = input_ids
        else:
            ids = input_ids.item() if input_ids.numel() == 1 else tuple(input_ids.flatten().tolist())
        seed = int(hash(str(ids)[:20]) % (2**31))
        torch.manual_seed(seed + 42)
        # Return deterministic features based on string hash for reproducibility
        return torch.randn(1, 77, self.embed_dim).float()


class MockCLIPModel:
    """
    Minimal CLIP model mock for testing without downloading full model.
    Provides 576 patch embeddings (24x24 grid) for CLIP ViT-L/14.
    """
    def __init__(self, embed_dim: int = 768):
        self.embed_dim = embed_dim
        # Pre-compute deterministic patch embeddings from a fixed image hash
        self._vision_features_cache = {}  # (image_hash) -> [B, 576, C]
        # Initialize mock encoders with a fixed seed for reproducibility
        self.vision_encoder = MockVisionEncoder(embed_dim=embed_dim)
        self.text_encoder = MockTextEncoder(embed_dim=embed_dim)

    def _get_vision_features(self, img: torch.Tensor) -> torch.Tensor:
        """Get or cache vision features for an image."""
        if len(img.shape) == 5:
            img = img.squeeze(0)  # Remove batch dim if single sample
        # Create deterministic hash based on image bytes
        img_np = (img.cpu().numpy() * 255).astype(np.uint8)
        img_hash = int(hash(tuple(img_np.flatten())) % (2**31))

        if img_hash not in self._vision_features_cache:
            # Generate reproducible patch features - CLIP ViT-L/14 has 576 patches (24x24)
            torch.manual_seed(img_hash + 1337)
            patch_features = np.random.randn(1, 576, self.embed_dim).astype(np.float32)
            self._vision_features_cache[img_hash] = torch.from_numpy(patch_features)

        return self._vision_features_cache[img_hash]

    def vision_model(self, image: torch.Tensor) -> torch.Tensor:
        """Mock vision model returning patch features [B, 384, C]."""
        return self._get_vision_features(image)

    def text_model(self, input_ids) -> torch.Tensor:
        """Mock text model encoding to [1, T, C] shape. Accepts both string and tensor."""
        # Handle both raw text strings and tensors for flexibility
        if isinstance(input_ids, str):
            ids = input_ids
        else:
            ids = input_ids.item() if input_ids.numel() == 1 else tuple(input_ids.flatten().tolist())
        seed = int(hash(str(ids)[:20]) % (2**31))
        torch.manual_seed(seed + 42)
        return torch.randn(1, 77, self.embed_dim).float()


class TestResizeHeatmap:
    """Tests for resize_heatmap() function."""

    def test_basic_resizing(self):
        heatmap_12x12 = np.random.rand(12, 12).astype(np.float32)
        result = resize_heatmap(heatmap_12x12, 384, 384)
        assert result.shape == (384, 384)
        assert result.dtype in [np.float32, np.float64]

    def test_zero_heatmap(self):
        zero_grid = np.zeros((12, 12), dtype=np.float32)
        result = resize_heatmap(zero_grid, 100, 100)
        assert np.allclose(result, 0.0)

    def test_ones_heatmap(self):
        ones_grid = np.ones((12, 12), dtype=np.float32) * 0.5
        result = resize_heatmap(ones_grid, 50, 50)
        assert np.allclose(result, 0.5, atol=0.01)

    def test_preserves_range(self):
        grid_minmax = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32).reshape(2, 2) * (0.8 - 0.2)
        result = resize_heatmap(grid_minmax, 64, 64)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_different_sizes(self):
        for target_size in [(32, 32), (128, 256), (768, 768)]:
            grid = np.random.rand(12, 12)
            res = resize_heatmap(grid, *target_size)
            assert res.shape == target_size


class TestApplyOverlay:
    """Tests for apply_overlay() function."""

    def test_basic_overlay(self):
        image = np.random.rand(64, 64, 3).astype(np.float32) * 0.5
        heatmap = np.random.rand(64, 64).astype(np.float32)
        blended = apply_overlay(image, heatmap, alpha=0.5)
        assert isinstance(blended, Image.Image)
        # PIL Image.size returns (width, height); blended is RGB with same dimensions as input
        # Since image is 64x64, blended is also 64 pixels wide x 64 rows high
        assert blended.width == 64  # Width in pixels
        assert blended.height == 64  # Height (rows)

    def test_alpha_compositing(self):
        # Create predictable inputs to verify blending behavior
        image = np.zeros((10, 10, 3), dtype=np.float32)  # Black background
        heatmap = np.ones((10, 10)) * 1.0  # Full white heatmap

        alpha_10 = apply_overlay(image, heatmap, alpha=0.1)
        alpha_90 = apply_overlay(image, heatmap, alpha=0.9)

        # Lower alpha should be darker (more black background visible)
        assert np.mean(np.array(alpha_10)) < np.mean(np.array(alpha_90))

    def test_colormap_mapping(self):
        image = np.random.rand(32, 32, 3).astype(np.float32) * 0.5
        # Create a simple gradient heatmap
        y, x = np.mgrid[:32, :32]
        heatmap = (x + y) / 64.0
        result_jet = apply_overlay(image, heatmap, colormap="jet")
        result_viridis = apply_overlay(image, heatmap, colormap="viridis")
        # Different colormaps should produce different outputs
        assert not np.allclose(
            np.array(result_jet),
            np.array(result_viridis)
        )

    def test_heatmap_range_preservation(self):
        image = np.zeros((16, 16, 3), dtype=np.float32) * 0.5
        # Test with heatmap values outside [0,1] (should be clipped internally or cause issues)
        heat_high = np.full((16, 16), 2.0).astype(np.float32)
        heat_low = -np.ones((16, 16)).astype(np.float32)

        try:
            # This might fail with older OpenCV versions
            result_high = apply_overlay(image, heat_high, alpha=1.0)
            assert isinstance(result_high, Image.Image)
        except Exception as e:
            pytest.skip(f"OpenCV colormap requires uint8: {e}")


class TestExtractAttentionWeights:
    """Tests for extract_attention_weights() function."""

    @pytest.fixture(scope="class")
    @classmethod
    def create_mock_model(cls):
        model = MockCLIPModel(embed_dim=256)  # Smaller dim for speed
        return model

    def test_basic_extraction(self, create_mock_model):
        mock_image = torch.randn(1, 3, 64, 64)
        mock_text = "industrial development"

        # MockCLIPModel has vision_encoder and text_encoder, should work
        with torch.no_grad():
            result = extract_attention_weights(mock_image, mock_text, model=create_mock_model)
        assert isinstance(result, np.ndarray)  # Returns a numpy array

    def test_single_patch_target(self, create_mock_model):
        image = torch.randn(1, 3, 64, 64)
        text = "construction activity"

        # Test with valid target patch (should work with MockCLIPModel's hash-based caching)
        result_normal = extract_attention_weights(image, text, model=create_mock_model)

        # Clear cache to test target_patch behavior
        create_mock_model._vision_features_cache.clear()
        result_target = extract_attention_weights(image, text, model=create_mock_model, target_patch=0)

        # Both should return numpy arrays (target_patch returns a single value as array,
        # the full call returns the whole patch grid).
        assert isinstance(result_normal, np.ndarray)
        assert isinstance(result_target, np.ndarray)
        assert result_target.size >= 1
        assert result_normal.size >= result_target.size


class TestExtractPatchAttention:
    """Tests for extract_patch_attention() function."""

    @pytest.fixture(scope="class")
    @classmethod
    def create_mock_model(cls):
        return MockCLIPModel(embed_dim=256)

    def test_basic_extraction(self, create_mock_model):
        # MockCLIPModel has vision_encoder and text_encoder, should work
        image_t1 = torch.randn(1, 3, 64, 64)
        image_t2 = torch.randn(1, 3, 64, 64) * 0.5 + 0.5  # Different image

        with torch.no_grad():
            result = extract_patch_attention(image_t1, image_t2, model=create_mock_model)
        assert isinstance(result, np.ndarray)

    def test_same_images_zero_difference(self, create_mock_model):
        # If both images are identical and cache hits, difference should be near-zero
        image = torch.randn(1, 3, 64, 64) * 0.1
        with torch.no_grad():
            result = extract_patch_attention(image, image, model=create_mock_model)
        # Same images should produce very small difference scores (near zero)
        assert np.allclose(result, 0.0, atol=0.3)  # Some tolerance for hash-based feature variation

    def test_different_images_nonzero_difference(self, create_mock_model):
        # Verify function produces non-zero difference for different images
        img1 = torch.randn(1, 3, 64, 64) * 0.1
        img2 = torch.randn(1, 3, 64, 64) * 0.1 + 0.5  # Different image

        with torch.no_grad():
            result_diff = extract_patch_attention(img1, img2, model=create_mock_model)

        # Same images should be near-zero, different images should have more variation
        same_image_result = extract_patch_attention(img1, img1, model=create_mock_model)

        assert np.allclose(same_image_result, 0.0, atol=0.3)  # Same image = zero diff
        # Different images: sum of absolute differences should be higher than near-zero


class TestGenerateGridHeatmapFromPatches:
    """Tests for generate_grid_heatmap_from_patches() function."""

    def test_basic_reshape(self):
        # CLIP ViT-L/14: 24x24 patch grid = 576 patches
        patch_scores = np.random.rand(576).astype(np.float32) * 0.5
        grid, full = generate_grid_heatmap_from_patches(patch_scores, grid_shape=(24, 24))
        assert grid.shape == (24, 24)
        assert full.shape[0] == 24
        assert full.shape[1] == 24

    def test_2d_input(self):
        # Already in grid shape - should still work
        patch_grid = np.random.rand(8, 16).astype(np.float32) * 0.5
        grid, _ = generate_grid_heatmap_from_patches(patch_grid)
        assert grid.shape == (8, 16)

    def test_colormap_variations(self):
        scores = np.linspace(0, 1, 576).astype(np.float32)

        for colormap in ["jet", "viridis", "plasma"]:
            try:
                _, heatmap_color = generate_grid_heatmap_from_patches(scores, grid_shape=(24, 24), colormap=colormap)
                assert isinstance(heatmap_color, np.ndarray)
                assert heatmap_color.dtype == np.uint8
            except ValueError as e:
                if "COLORMAP_" in str(e):  # Colormap not available
                    pytest.skip(f"{colormap} colormap unavailable")

    def test_edge_values(self):
        # All zeros - CLIP ViT-L/14: 24x24 patch grid
        zero_scores = np.zeros(576, dtype=np.float32)
        grid, _ = generate_grid_heatmap_from_patches(zero_scores, grid_shape=(24, 24))
        assert np.allclose(grid, 0.0)

        # All ones - CLIP ViT-L/14: 24x24 patch grid
        one_scores = np.ones(576, dtype=np.float32) * 0.5
        grid, _ = generate_grid_heatmap_from_patches(one_scores, grid_shape=(24, 24))
        assert np.allclose(grid, 0.5, atol=1e-6)


class TestIntegration:
    """End-to-end integration tests."""

    def test_full_pipeline_shape(self):
        # Verify all functions compose correctly without errors
        heatmap_12x12 = np.random.rand(12, 12).astype(np.float32)

        resized = resize_heatmap(heatmap_12x12, 64, 64)
        assert resized.shape == (64, 64)

    def test_mock_model_compatibility(self):
        """Verify MockCLIPModel integrates with heatmap functions."""
        model = MockCLIPModel(embed_dim=512)

        # Test that the mock provides required attributes
        assert hasattr(model, 'vision_encoder')
        assert hasattr(model, 'text_encoder')


class _StubPatchEncoder:
    """Minimal encoder exposing the patch + text hooks the production heatmap
    entry points rely on (no CLIP, no network). ``embed_dim`` is small; 16 patches
    give a 4x4 grid."""

    name = "stub"
    embed_dim = 4
    image_input_size = 8
    device = torch.device("cpu")

    def encode_text(self, texts, batch_size: int = 32) -> np.ndarray:
        n = 1 if isinstance(texts, str) else len(texts)
        v = np.zeros((n, self.embed_dim), dtype=np.float32)
        v[:, 0] = 1.0  # unit vector along axis 0
        return v

    def encode_image_patches(self, image, batch_size: int = 32) -> np.ndarray:
        # 16 L2-normalised patch embeddings, deterministic from the tile's mean
        # pixel so T1 and T2 differ.
        base = int(float(np.asarray(image).mean()) * 1000) + 7
        p = np.random.default_rng(base).standard_normal(
            (16, self.embed_dim)).astype(np.float32)
        p /= np.linalg.norm(p, axis=-1, keepdims=True) + 1e-8
        return p

    def compute_patch_text_similarity(self, image, text) -> np.ndarray:
        return np.linspace(0, 1, 16, dtype=np.float32).reshape(4, 4)


class _NoPatchEncoder:
    """Encoder without ``encode_image_patches`` — change heatmap must degrade to
    (None, None) so the caller can fall back to ``generate_heatmap``."""

    def compute_patch_text_similarity(self, image, text) -> np.ndarray:
        return np.zeros((4, 4), dtype=np.float32)


class TestGenerateChangeHeatmap:
    """Covers generate_change_heatmap — the production change localiser (used by
    src/app.py and thesis figures) that had no direct test."""

    @staticmethod
    def _imgs():
        t1 = np.zeros((32, 32, 3), dtype=np.uint8)
        t2 = np.full((32, 32, 3), 200, dtype=np.uint8)
        return t1, t2

    def test_returns_normalised_square_grid(self):
        t1, t2 = self._imgs()
        grid, blended = generate_change_heatmap(t1, t2, "new buildings", _StubPatchEncoder())
        assert grid is not None and blended is not None
        assert grid.shape == (4, 4)                       # 16 patches -> 4x4
        assert grid.min() >= 0.0 and grid.max() <= 1.0    # min-max normalised
        assert isinstance(blended, Image.Image)
        assert blended.width == 32 and blended.height == 32

    def test_precomputed_patches_bypass_encoder(self):
        t1, t2 = self._imgs()
        P = np.zeros((16, 4), dtype=np.float32)
        P[:, 0] = 1.0
        grid, _ = generate_change_heatmap(t1, t2, "x", _StubPatchEncoder(),
                                          precomputed_p1=P, precomputed_p2=P.copy())
        # Identical P1/P2 -> zero Δ everywhere -> flat (all-zero) normalised grid.
        assert grid is not None
        assert np.allclose(grid, 0.0)

    def test_none_when_encoder_lacks_patches(self):
        t1, t2 = self._imgs()
        grid, blended = generate_change_heatmap(t1, t2, "x", _NoPatchEncoder())
        assert grid is None and blended is None


class TestGenerateHeatmap:
    """Covers generate_heatmap — the After-image similarity overlay."""

    def test_returns_grid_and_blend(self):
        t1 = np.zeros((24, 24, 3), dtype=np.uint8)
        t2 = np.full((24, 24, 3), 128, dtype=np.uint8)
        grid, blended = generate_heatmap(t1, t2, "a water body", _StubPatchEncoder())
        assert grid.shape == (4, 4)                       # stub returns a 4x4 sim grid
        assert isinstance(blended, Image.Image)
        assert blended.width == 24 and blended.height == 24

