"""
Behavioural tests for the PEFT training loss `_masked_infonce` (train.py) — the
loss that actually trains the ProjectionHead adapter behind the report §8.2 PEFT numbers.

These assert robust properties (finite, non-negative, lower when aligned), not
exact values, so they don't depend on the temperature constant.
"""
import torch

from src.train import _masked_infonce


def test_masked_infonce_lower_when_aligned():
    """Aligned proj/text (diagonal positives) must score far below anti-aligned."""
    torch.manual_seed(0)
    B, D = 4, 16
    text = torch.randn(B, D)
    pos = torch.eye(B, dtype=torch.bool)  # each pair is its own positive

    aligned = _masked_infonce(text.clone(), text.clone(), pos)
    anti = _masked_infonce(-text.clone(), text.clone(), pos)

    assert torch.isfinite(aligned) and aligned.item() >= 0.0
    assert aligned.item() < anti.item()


def test_masked_infonce_multi_positive_mask_is_finite():
    """Multi-positive mask (same-caption rows share positives) yields a finite loss."""
    torch.manual_seed(1)
    B, D = 4, 16
    text = torch.randn(B, D)
    pos = torch.tensor(
        [[1, 1, 0, 0],
         [1, 1, 0, 0],
         [0, 0, 1, 1],
         [0, 0, 1, 1]],
        dtype=torch.bool,
    )
    loss = _masked_infonce(text.clone(), text.clone(), pos)
    assert torch.isfinite(loss) and loss.item() >= 0.0


def test_masked_infonce_is_symmetric_scalar():
    """Returns a scalar; symmetric in construction (0.5*(dir + dir.t()))."""
    torch.manual_seed(2)
    B, D = 3, 8
    proj = torch.randn(B, D)
    text = torch.randn(B, D)
    pos = torch.eye(B, dtype=torch.bool)
    loss = _masked_infonce(proj, text, pos)
    assert loss.ndim == 0 and torch.isfinite(loss)
