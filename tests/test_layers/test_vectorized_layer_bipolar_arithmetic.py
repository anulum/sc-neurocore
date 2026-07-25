# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — VectorizedSCLayer bipolar arithmetic contracts

"""Verify signed packed-XNOR arithmetic and its input domain."""

import numpy as np
import pytest

from sc_neurocore.layers.vectorized_layer import VectorizedSCLayer


def test_vectorized_bipolar_dense_preserves_signed_dot_product() -> None:
    """Negative weights contribute negative terms under packed XNOR arithmetic."""
    np.random.seed(0)
    layer = VectorizedSCLayer(
        n_inputs=2, n_neurons=1, length=4096, use_gpu=False, sc_mode="bipolar"
    )
    layer.weights[:] = np.array([[1.0, -1.0]])
    layer._refresh_packed_weights()

    assert abs(layer.forward([1.0, 1.0])[0]) < 0.02


def test_vectorized_bipolar_dense_handles_fractional_signed_weights() -> None:
    """Packed bipolar output approximates a fractional signed dot product."""
    np.random.seed(1)
    layer = VectorizedSCLayer(
        n_inputs=2, n_neurons=1, length=65536, use_gpu=False, sc_mode="bipolar"
    )
    layer.weights[:] = np.array([[0.5, -0.25]])
    layer._refresh_packed_weights()

    assert np.allclose(layer.forward([1.0, -1.0]), [0.75], atol=0.03)


def test_vectorized_bipolar_rejects_out_of_range_input() -> None:
    """Bipolar mode accepts only signed values in [-1, 1]."""
    layer = VectorizedSCLayer(n_inputs=2, n_neurons=1, length=64, use_gpu=False, sc_mode="bipolar")
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        layer.forward([0.0, 1.5])
