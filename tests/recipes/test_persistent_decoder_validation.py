"""Exact-format error bounds used to qualify persistent decoder arithmetic."""

import pytest
import torch

from ignition.recipes.deepseek41.persistent_validation import (
    _dot,
    _fp8_spacing,
    _quantized,
)


def test_e4m3_spacing_uses_format_bits_at_binade_boundaries():
    values = torch.tensor(
        [0, 2.0**-9, 2.0**-6, 1, 2, 16, 64, 128, 256, 448, -64],
        dtype=torch.float8_e4m3fn,
    )
    expected = torch.tensor(
        [2.0**-9, 2.0**-9, 2.0**-9, 0.125, 0.25, 2, 8, 16, 32, 32, 8],
        dtype=torch.float64,
    )
    torch.testing.assert_close(_fp8_spacing(values), expected, rtol=0, atol=0)


def test_dense_quantizer_keeps_unfloored_small_blocks_and_zero_scale_one():
    source = (
        torch.cat((torch.linspace(0, 1e-9, 32), torch.zeros(32))).bfloat16().view(1, 64)
    )
    values, scales, _ = _quantized(source)
    assert int(scales[0, 0]) == 89 and int(scales[0, 1]) == 127
    assert int(torch.count_nonzero(values.view(torch.uint8)[0, :32])) > 0


def test_fp64_dot_bound_rejects_values_outside_rounding_bound():
    left = torch.tensor([[1.0, 2.0, -0.5]], dtype=torch.float64)
    right = torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float64)
    actual = (left @ right.T).float().bfloat16()
    _dot(actual, left, right, role="bounded test projection")
    with pytest.raises(AssertionError, match="declared rounding bounds"):
        _dot(
            actual + torch.tensor(0.125, dtype=torch.bfloat16),
            left,
            right,
            role="bounded test projection",
        )


def test_rotary_projection_requires_zero_output_for_inactive_rows():
    from ignition.recipes.deepseek41.persistent_validation import _rotated_dot

    left = torch.ones((2, 1), dtype=torch.float64)
    right = torch.full((512, 1), 0.125, dtype=torch.float64)
    angles = torch.zeros((2, 32, 2))
    angles[..., 0] = 1
    positions = torch.tensor([16, -1])
    output = torch.full((2, 1, 512), 0.125, dtype=torch.bfloat16)
    output[1] = 0
    _rotated_dot(output, left, right, angles, positions, role="masked rotary projection")
    output[1, 0, 0] = 0.125
    with pytest.raises(AssertionError, match="declared rounding bounds"):
        _rotated_dot(output, left, right, angles, positions, role="masked rotary projection")
