# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SC convolution configuration tests

"""Kernel construction, deterministic seeding, and configuration contracts."""

import time
from typing import Any

import numpy as np
import pytest

from sc_neurocore.layers.sc_conv_layer import SCConv2DLayer
from tests.layers.sc_conv_layer_support import _perf_enabled


def test_conv_kernel_shape() -> None:
    """Kernels should match (out, in, k, k)."""
    np.random.seed(0)
    layer = SCConv2DLayer(in_channels=2, out_channels=3, kernel_size=3)
    assert layer.kernels.shape == (3, 2, 3, 3)


def test_conv_deterministic_with_seed() -> None:
    """Setting numpy seed produces repeatable kernels."""
    np.random.seed(42)
    layer_a = SCConv2DLayer(in_channels=1, out_channels=1, kernel_size=3)
    np.random.seed(42)
    layer_b = SCConv2DLayer(in_channels=1, out_channels=1, kernel_size=3)
    assert np.allclose(layer_a.kernels, layer_b.kernels)


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


@pytest.mark.skipif(not _perf_enabled(), reason="Set SC_NEUROCORE_PERF=1 to enable perf checks.")
def test_conv_layer_perf_small() -> None:
    """Benchmark a tiny convolution for performance sanity."""
    layer = SCConv2DLayer(in_channels=1, out_channels=1, kernel_size=3)
    inp = np.random.random((1, 16, 16))
    start = time.perf_counter()
    _ = layer.forward(inp)
    elapsed = time.perf_counter() - start
    assert elapsed < 2.0
