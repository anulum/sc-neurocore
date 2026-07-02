# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for SCConv2DLayer output shapes and edge cases

"""Tests for SCConv2DLayer output shapes and edge cases."""

import os
import time
from typing import Any

import numpy as np
import pytest

from sc_neurocore.layers.sc_conv_layer import SCConv2DLayer


def _perf_enabled() -> bool:
    return os.environ.get("SC_NEUROCORE_PERF") == "1"


def test_conv_kernel_shape() -> None:
    """Kernels should match (out, in, k, k)."""
    np.random.seed(0)
    layer = SCConv2DLayer(in_channels=2, out_channels=3, kernel_size=3)
    assert layer.kernels.shape == (3, 2, 3, 3)


def test_conv_output_shape_no_padding() -> None:
    """Output shape follows convolution formula without padding."""
    layer = SCConv2DLayer(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=0)
    inp = np.ones((1, 5, 5))
    out = layer.forward(inp)
    assert out.shape == (2, 3, 3)


def test_conv_output_shape_with_padding_and_stride() -> None:
    """Output shape follows convolution formula with padding and stride."""
    layer = SCConv2DLayer(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1)
    inp = np.ones((1, 6, 6))
    out = layer.forward(inp)
    assert out.shape == (1, 3, 3)


def test_conv_forward_zero_input() -> None:
    """Zero input should produce all-zero output."""
    layer = SCConv2DLayer(in_channels=1, out_channels=1, kernel_size=3)
    inp = np.zeros((1, 5, 5))
    out = layer.forward(inp)
    assert np.allclose(out, 0.0)


def test_conv_forward_known_kernel() -> None:
    """Known kernel and input yield deterministic sum output."""
    layer = SCConv2DLayer(in_channels=2, out_channels=1, kernel_size=2)
    layer.kernels[:] = 1.0
    inp = np.ones((2, 3, 3))
    out = layer.forward(inp)
    # Each 2x2 region sum = 4 per channel, 2 channels -> 8
    assert np.allclose(out, 8.0)


def test_conv_bipolar_signed_kernel_and_input() -> None:
    """Bipolar mode supports signed XNOR-equivalent products."""
    layer = SCConv2DLayer(in_channels=1, out_channels=1, kernel_size=2, sc_mode="bipolar")
    layer.kernels[:] = np.array([[[[1.0, -1.0], [0.5, -0.5]]]])
    inp = np.array([[[1.0, -1.0], [-1.0, 1.0]]])
    out = layer.forward(inp)
    expected = np.array([[[1.0 + 1.0 - 0.5 - 0.5]]])
    assert np.allclose(out, expected)


def test_conv_deterministic_with_seed() -> None:
    """Setting numpy seed produces repeatable kernels."""
    np.random.seed(42)
    layer_a = SCConv2DLayer(in_channels=1, out_channels=1, kernel_size=3)
    np.random.seed(42)
    layer_b = SCConv2DLayer(in_channels=1, out_channels=1, kernel_size=3)
    assert np.allclose(layer_a.kernels, layer_b.kernels)


def test_conv_input_channel_mismatch_raises() -> None:
    """Mismatched input channels should raise an indexing error."""
    layer = SCConv2DLayer(in_channels=2, out_channels=1, kernel_size=3)
    inp = np.ones((1, 5, 5))
    with pytest.raises(IndexError):
        layer.forward(inp)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"in_channels": 0, "out_channels": 1, "kernel_size": 3},
        {"in_channels": 1, "out_channels": 0, "kernel_size": 3},
        {"in_channels": 1, "out_channels": 1, "kernel_size": 0},
        {"in_channels": 1, "out_channels": 1, "kernel_size": 3, "stride": 0},
        {"in_channels": 1, "out_channels": 1, "kernel_size": 3, "padding": -1},
        {"in_channels": 1, "out_channels": 1, "kernel_size": 3, "length": 0},
        {"in_channels": 1, "out_channels": 1, "kernel_size": 3, "sc_mode": "ternary"},
    ],
)
def test_conv_invalid_configuration_raises(kwargs: dict[str, Any]) -> None:
    """Invalid convolution configuration should fail at construction."""
    with pytest.raises(ValueError):
        SCConv2DLayer(**kwargs)


def test_conv_unipolar_rejects_out_of_range_input() -> None:
    """Unipolar mode should reject invalid probability values."""
    layer = SCConv2DLayer(in_channels=1, out_channels=1, kernel_size=1)
    with pytest.raises(ValueError, match="unipolar"):
        layer.forward(np.array([[[1.01]]]))


def test_conv_bipolar_rejects_out_of_range_input() -> None:
    """Bipolar mode should reject values outside [-1, 1]."""
    layer = SCConv2DLayer(in_channels=1, out_channels=1, kernel_size=1, sc_mode="bipolar")
    with pytest.raises(ValueError, match="bipolar"):
        layer.forward(np.array([[[-1.01]]]))


def test_conv_rejects_empty_output_geometry() -> None:
    """Kernels larger than the padded image should not produce silent empty output."""
    layer = SCConv2DLayer(in_channels=1, out_channels=1, kernel_size=5)
    with pytest.raises(ValueError, match="empty output"):
        layer.forward(np.ones((1, 3, 3)))


def test_conv_rejects_non_finite_input() -> None:
    """NaN and infinity are invalid stochastic probabilities."""
    layer = SCConv2DLayer(in_channels=1, out_channels=1, kernel_size=1)
    with pytest.raises(ValueError, match="finite"):
        layer.forward(np.array([[[np.nan]]]))


def test_conv_rejects_rank_mismatch() -> None:
    """Input tensors must be channel-first images."""
    layer = SCConv2DLayer(in_channels=1, out_channels=1, kernel_size=1)
    with pytest.raises(ValueError, match="shape"):
        layer.forward(np.ones((3, 3)))


def test_conv_padding_changes_output_size() -> None:
    """Padding should expand the output grid compared to no padding."""
    base = SCConv2DLayer(in_channels=1, out_channels=1, kernel_size=3, padding=0)
    padded = SCConv2DLayer(in_channels=1, out_channels=1, kernel_size=3, padding=1)
    inp = np.ones((1, 5, 5))
    assert padded.forward(inp).shape[1] > base.forward(inp).shape[1]


def test_conv_stride_changes_output_size() -> None:
    """Stride should reduce output resolution."""
    stride1 = SCConv2DLayer(in_channels=1, out_channels=1, kernel_size=3, stride=1)
    stride2 = SCConv2DLayer(in_channels=1, out_channels=1, kernel_size=3, stride=2)
    inp = np.ones((1, 7, 7))
    assert stride2.forward(inp).shape[1] < stride1.forward(inp).shape[1]


@pytest.mark.skipif(not _perf_enabled(), reason="Set SC_NEUROCORE_PERF=1 to enable perf checks.")
def test_conv_layer_perf_small() -> None:
    """Benchmark a tiny convolution for performance sanity."""
    layer = SCConv2DLayer(in_channels=1, out_channels=1, kernel_size=3)
    inp = np.random.random((1, 16, 16))
    start = time.perf_counter()
    _ = layer.forward(inp)
    elapsed = time.perf_counter() - start
    assert elapsed < 2.0


def test_conv_rejects_nonpositive_spatial_dimensions() -> None:
    layer = SCConv2DLayer(in_channels=1, out_channels=2, kernel_size=3)
    # A zero-height image clears the channel check but fails the positive-size guard.
    with pytest.raises(ValueError, match="height and width must be positive"):
        layer.forward(np.zeros((1, 0, 5), dtype=np.float64))
