# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Public packed layer multiply-accumulate output contracts

"""Preserve full exported-weight MAC outputs beyond firing-fraction bounds."""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.layers.vectorized_layer import VectorizedSCLayer


@pytest.mark.parametrize("length", [8, 65])
@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize(
    ("mode", "weights", "inputs", "expected"),
    [
        ("unipolar", [[1.0, 1.0, 1.0], [0.0, 1.0, 0.0]], [1.0, 1.0, 1.0], [3.25, 0.75]),
        ("bipolar", [[1.0, -1.0, 1.0], [-1.0, -1.0, -1.0]], [1.0, -1.0, 1.0], [3.25, -1.25]),
    ],
)
def test_exported_endpoint_mac_outputs_are_unclipped_channel_sums(
    mode: str,
    weights: list[list[float]],
    inputs: list[float],
    expected: list[float],
    sparse: bool,
    length: int,
) -> None:
    """Match the complete endpoint dot product without relabelling it as spike density.

    Parameters
    ----------
    mode
        Actual unipolar or bipolar encoding selected by the exported layer.
    weights
        Full endpoint weight matrix, whose packed products are exact.
    inputs
        Full endpoint input vector in the selected encoding's domain.
    expected
        Independently calculated complete dot-product outputs including bias.
    sparse
        Select the existing public dense or CSR realization.
    length
        Meaningful bit length, including a partial final packed word.
    """
    matrix = np.asarray(weights, dtype=np.float64)
    values = np.asarray(inputs, dtype=np.float64)
    bias = np.array([0.25, -0.25], dtype=np.float64)
    for array in (matrix, values, bias):
        array.setflags(write=False)
    before = [array.tobytes() for array in (matrix, values, bias)]
    layer = VectorizedSCLayer.from_exported_weights(
        {"weight": matrix, "bias": bias, "encoding": mode},
        use_gpu=False,
        sparse=sparse,
        length=length,
        seed=42,
    )
    output = layer.forward(values)
    assert output.shape == (2,) and output.dtype == np.float64
    np.testing.assert_array_equal(output, expected)
    np.testing.assert_array_equal(output, matrix @ values + bias)
    np.testing.assert_array_equal(layer.forward(values), expected)
    assert output[0] > 1.0
    if mode == "bipolar":
        assert output[1] < 0.0
    assert [array.tobytes() for array in (matrix, values, bias)] == before
    assert all(not array.flags.writeable for array in (matrix, values, bias))
