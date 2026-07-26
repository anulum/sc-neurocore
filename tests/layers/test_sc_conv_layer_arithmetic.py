# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SC convolution arithmetic tests

"""Zero, known-kernel, and bipolar signed arithmetic contracts."""

import numpy as np

from sc_neurocore.layers.sc_conv_layer import SCConv2DLayer


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
