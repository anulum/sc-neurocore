"""Tests for SCConv2DLayer output shapes and edge cases."""

import os
import time

import numpy as np
import pytest

from sc_neurocore.layers.sc_conv_layer import SCConv2DLayer


def _perf_enabled() -> bool:
    return os.environ.get("SC_NEUROCORE_PERF") == "1"


def test_conv_kernel_shape():
    """Kernels should match (out, in, k, k)."""
    np.random.seed(0)
    layer = SCConv2DLayer(in_channels=2, out_channels=3, kernel_size=3)
    assert layer.kernels.shape == (3, 2, 3, 3)


def test_conv_output_shape_no_padding():
    """Output shape follows convolution formula without padding."""
    layer = SCConv2DLayer(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=0)
    inp = np.ones((1, 5, 5))
    out = layer.forward(inp)
    assert out.shape == (2, 3, 3)


def test_conv_output_shape_with_padding_and_stride():
    """Output shape follows convolution formula with padding and stride."""
    layer = SCConv2DLayer(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1)
    inp = np.ones((1, 6, 6))
    out = layer.forward(inp)
    assert out.shape == (1, 3, 3)


def test_conv_forward_zero_input():
    """Zero input should produce all-zero output."""
    layer = SCConv2DLayer(in_channels=1, out_channels=1, kernel_size=3)
    inp = np.zeros((1, 5, 5))
    out = layer.forward(inp)
    assert np.allclose(out, 0.0)


def test_conv_forward_known_kernel():
    """Known kernel and input yield deterministic sum output."""
    layer = SCConv2DLayer(in_channels=2, out_channels=1, kernel_size=2)
    layer.kernels[:] = 1.0
    inp = np.ones((2, 3, 3))
    out = layer.forward(inp)
    # Each 2x2 region sum = 4 per channel, 2 channels -> 8
    assert np.allclose(out, 8.0)


def test_conv_deterministic_with_seed():
    """Setting numpy seed produces repeatable kernels."""
    np.random.seed(42)
    layer_a = SCConv2DLayer(in_channels=1, out_channels=1, kernel_size=3)
    np.random.seed(42)
    layer_b = SCConv2DLayer(in_channels=1, out_channels=1, kernel_size=3)
    assert np.allclose(layer_a.kernels, layer_b.kernels)


def test_conv_input_channel_mismatch_raises():
    """Mismatched input channels should raise an indexing error."""
    layer = SCConv2DLayer(in_channels=2, out_channels=1, kernel_size=3)
    inp = np.ones((1, 5, 5))
    with pytest.raises(IndexError):
        layer.forward(inp)


def test_conv_padding_changes_output_size():
    """Padding should expand the output grid compared to no padding."""
    base = SCConv2DLayer(in_channels=1, out_channels=1, kernel_size=3, padding=0)
    padded = SCConv2DLayer(in_channels=1, out_channels=1, kernel_size=3, padding=1)
    inp = np.ones((1, 5, 5))
    assert padded.forward(inp).shape[1] > base.forward(inp).shape[1]


def test_conv_stride_changes_output_size():
    """Stride should reduce output resolution."""
    stride1 = SCConv2DLayer(in_channels=1, out_channels=1, kernel_size=3, stride=1)
    stride2 = SCConv2DLayer(in_channels=1, out_channels=1, kernel_size=3, stride=2)
    inp = np.ones((1, 7, 7))
    assert stride2.forward(inp).shape[1] < stride1.forward(inp).shape[1]


@pytest.mark.skipif(not _perf_enabled(), reason="Set SC_NEUROCORE_PERF=1 to enable perf checks.")
def test_conv_layer_perf_small():
    """Benchmark a tiny convolution for performance sanity."""
    layer = SCConv2DLayer(in_channels=1, out_channels=1, kernel_size=3)
    inp = np.random.random((1, 16, 16))
    start = time.perf_counter()
    _ = layer.forward(inp)
    elapsed = time.perf_counter() - start
    assert elapsed < 2.0
