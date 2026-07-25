# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — VectorizedSCLayer packed-forward contracts

"""Verify packed representation, unipolar forward behavior, and its opt-in budget."""

import time

import numpy as np
import pytest

from sc_neurocore.layers.vectorized_layer import VectorizedSCLayer
from tests.test_layers.vectorized_layer_support import _expected_words, _perf_enabled


def test_vectorized_packed_shape() -> None:
    """Packed weights should have the expected word dimension."""
    np.random.seed(0)
    layer = VectorizedSCLayer(n_inputs=3, n_neurons=2, length=130)
    assert layer.packed_weights.shape == (2, 3, _expected_words(130))


def test_vectorized_packed_dtype() -> None:
    """Packed weights should be uint64 for bitwise operations."""
    layer = VectorizedSCLayer(n_inputs=2, n_neurons=2, length=64)
    assert layer.packed_weights.dtype == np.uint64


def test_vectorized_forward_shape() -> None:
    """Forward returns one output per neuron."""
    layer = VectorizedSCLayer(n_inputs=2, n_neurons=4, length=32)
    assert layer.forward([0.3, 0.7]).shape == (4,)


def test_vectorized_forward_zero_input_returns_zero() -> None:
    """Zero inputs yield zero outputs."""
    layer = VectorizedSCLayer(n_inputs=3, n_neurons=2, length=32)
    assert np.allclose(layer.forward([0.0, 0.0, 0.0]), 0.0)


def test_vectorized_output_range() -> None:
    """Unipolar outputs stay within zero and the input count."""
    layer = VectorizedSCLayer(n_inputs=4, n_neurons=3, length=64)
    output = layer.forward([0.2, 0.4, 0.6, 0.8])
    assert np.all(output >= 0.0)
    assert np.all(output <= 4.0)


def test_vectorized_refresh_changes_packed_weights() -> None:
    """Refreshing after a weight change updates the packed representation."""
    np.random.seed(1)
    layer = VectorizedSCLayer(n_inputs=2, n_neurons=2, length=32)
    before = layer.packed_weights.copy()
    layer.weights[:] = 0.0
    layer._refresh_packed_weights()
    assert not np.array_equal(before, layer.packed_weights)


def test_vectorized_deterministic_with_seed() -> None:
    """Setting the NumPy seed yields repeatable initial weights."""
    np.random.seed(99)
    first = VectorizedSCLayer(n_inputs=2, n_neurons=2, length=32)
    np.random.seed(99)
    second = VectorizedSCLayer(n_inputs=2, n_neurons=2, length=32)
    assert np.allclose(first.weights, second.weights)


def test_vectorized_input_length_mismatch_raises() -> None:
    """Forward rejects an input vector with the wrong length."""
    layer = VectorizedSCLayer(n_inputs=3, n_neurons=2, length=16)
    with pytest.raises(ValueError):
        layer.forward([0.1, 0.2])


def test_vectorized_length_not_multiple_of_64() -> None:
    """Non-word-aligned bitstream lengths remain runnable."""
    layer = VectorizedSCLayer(n_inputs=1, n_neurons=1, length=70)
    assert layer.forward([0.5]).shape == (1,)


@pytest.mark.skipif(not _perf_enabled(), reason="Set SC_NEUROCORE_PERF=1 to enable perf checks.")
def test_vectorized_layer_perf_small() -> None:
    """Keep a small packed forward pass within its opt-in smoke budget."""
    layer = VectorizedSCLayer(n_inputs=8, n_neurons=32, length=128)
    start = time.perf_counter()
    layer.forward([0.5] * 8)
    assert time.perf_counter() - start < 3.0
