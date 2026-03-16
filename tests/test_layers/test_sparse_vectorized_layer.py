# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for VectorizedSCLayer sparse (CSR) path

"""Tests for VectorizedSCLayer sparse (CSR) path."""

import numpy as np
import pytest

from sc_neurocore.layers.vectorized_layer import VectorizedSCLayer


@pytest.fixture(autouse=True)
def _seed():
    np.random.seed(42)


def test_sparse_dense_equivalence():
    """sparse(connectivity=1.0) must match dense output exactly."""
    np.random.seed(7)
    dense = VectorizedSCLayer(n_inputs=4, n_neurons=3, length=256, use_gpu=False)
    dense_weights = dense.weights.copy()
    dense_packed = dense.packed_weights.copy()

    np.random.seed(7)
    sparse = VectorizedSCLayer(
        n_inputs=4, n_neurons=3, length=256, use_gpu=False, sparse=True, connectivity=1.0
    )
    # Overwrite sparse weights and packed data to match dense exactly
    import scipy.sparse as sp

    sparse.weights_csr = sp.csr_matrix(dense_weights)
    sparse._sparse_packed = dense_packed.reshape(-1, dense_packed.shape[-1])

    np.random.seed(99)
    out_dense = dense.forward([0.3, 0.5, 0.7, 0.9])
    np.random.seed(99)
    out_sparse = sparse.forward([0.3, 0.5, 0.7, 0.9])

    np.testing.assert_array_equal(out_dense, out_sparse)


def test_sparse_memory_reduction():
    """At 10% connectivity, sparse storage uses <20% of dense memory."""
    dense = VectorizedSCLayer(n_inputs=100, n_neurons=100, length=256, use_gpu=False)
    sparse = VectorizedSCLayer(
        n_inputs=100, n_neurons=100, length=256, use_gpu=False, sparse=True, connectivity=0.1
    )

    dense_bytes = dense.packed_weights.nbytes
    sparse_bytes = sparse._sparse_packed.nbytes + sparse.weights_csr.data.nbytes
    assert sparse_bytes < 0.20 * dense_bytes


def test_sparse_forward_shape():
    """Output shape matches (n_neurons,) regardless of sparsity."""
    layer = VectorizedSCLayer(
        n_inputs=8, n_neurons=5, length=128, use_gpu=False, sparse=True, connectivity=0.5
    )
    out = layer.forward([0.5] * 8)
    assert out.shape == (5,)
    assert out.dtype == np.float64


def test_sparse_10pct_connectivity():
    """10% connectivity produces valid non-negative outputs."""
    layer = VectorizedSCLayer(
        n_inputs=20, n_neurons=10, length=256, use_gpu=False, sparse=True, connectivity=0.1
    )
    out = layer.forward(np.random.uniform(0, 1, 20).tolist())
    assert np.all(out >= 0.0)
    assert np.all(np.isfinite(out))


def test_sparse_1pct_connectivity():
    """1% connectivity produces valid outputs (many zeros expected)."""
    layer = VectorizedSCLayer(
        n_inputs=50, n_neurons=50, length=256, use_gpu=False, sparse=True, connectivity=0.01
    )
    out = layer.forward(np.random.uniform(0, 1, 50).tolist())
    assert out.shape == (50,)
    assert np.all(out >= 0.0)


def test_sparse_performance_scaling():
    """1000-neuron layer at 10% connectivity runs without OOM."""
    layer = VectorizedSCLayer(
        n_inputs=1000, n_neurons=1000, length=256, use_gpu=False, sparse=True, connectivity=0.1
    )
    out = layer.forward(np.random.uniform(0, 1, 1000).tolist())
    assert out.shape == (1000,)
    assert np.all(np.isfinite(out))


def test_sparse_zero_input_returns_zero():
    """Zero-valued inputs yield zero outputs in sparse mode."""
    layer = VectorizedSCLayer(
        n_inputs=4, n_neurons=3, length=128, use_gpu=False, sparse=True, connectivity=0.5
    )
    out = layer.forward([0.0, 0.0, 0.0, 0.0])
    assert np.allclose(out, 0.0)


def test_sparse_connectivity_validation():
    """connectivity outside (0, 1] raises ValueError."""
    with pytest.raises(ValueError, match="connectivity"):
        VectorizedSCLayer(n_inputs=4, n_neurons=3, sparse=True, connectivity=0.0)
    with pytest.raises(ValueError, match="connectivity"):
        VectorizedSCLayer(n_inputs=4, n_neurons=3, sparse=True, connectivity=1.5)


def test_sparse_input_mismatch_raises():
    """Wrong input length raises ValueError in sparse mode."""
    layer = VectorizedSCLayer(
        n_inputs=4, n_neurons=3, length=64, use_gpu=False, sparse=True, connectivity=0.5
    )
    with pytest.raises(ValueError):
        layer.forward([0.1, 0.2])


def test_dense_path_unchanged():
    """sparse=False (default) still works identically."""
    layer = VectorizedSCLayer(n_inputs=3, n_neurons=2, length=64, use_gpu=False)
    assert not layer.sparse
    out = layer.forward([0.3, 0.5, 0.7])
    assert out.shape == (2,)
    assert np.all(out >= 0.0)
