# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SC convolution geometry tests

"""Output shape, padding, stride, and spatial-dimension contracts."""

import numpy as np
import pytest

from sc_neurocore.layers.sc_conv_layer import SCConv2DLayer


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


def test_conv_rejects_nonpositive_spatial_dimensions() -> None:
    layer = SCConv2DLayer(in_channels=1, out_channels=2, kernel_size=3)
    # A zero-height image clears the channel check but fails the positive-size guard.
    with pytest.raises(ValueError, match="height and width must be positive"):
        layer.forward(np.zeros((1, 0, 5), dtype=np.float64))
