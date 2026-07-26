# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SC convolution input validation tests

"""Channel, range, geometry, finiteness, and rank validation contracts."""

import numpy as np
import pytest

from sc_neurocore.layers.sc_conv_layer import SCConv2DLayer


def test_conv_input_channel_mismatch_raises() -> None:
    """Mismatched input channels should raise an indexing error."""
    layer = SCConv2DLayer(in_channels=2, out_channels=1, kernel_size=3)
    inp = np.ones((1, 5, 5))
    with pytest.raises(IndexError):
        layer.forward(inp)


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
