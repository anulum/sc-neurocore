# SPDX-License-Identifier: AGPL-3.0-or-later
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
"""PyO3 round-trip tests for sparse CSR graph backend."""

import numpy as np
import pytest

sc = pytest.importorskip("sc_neurocore_engine", exc_type=ImportError)


def test_from_sparse_forward():
    # Ring graph: 0→1, 1→2, 2→0
    row_offsets = [0, 1, 2, 3]
    col_indices = [1, 2, 0]
    values = [1.0, 1.0, 1.0]
    layer = sc.StochasticGraphLayer.from_sparse(row_offsets, col_indices, values, 3, 2, seed=42)
    assert layer.is_sparse()
    features = np.random.default_rng(1).random((3, 2))
    out = np.array(layer.forward(features))
    assert out.shape == (3, 2)
    assert np.all(np.isfinite(out))


def test_from_dense_auto_sparse():
    # 10x10 with 5 edges → 5% density → should auto-select sparse
    adj = np.zeros((10, 10))
    adj[0, 1] = adj[1, 2] = adj[2, 3] = adj[3, 4] = adj[4, 0] = 1.0
    layer = sc.StochasticGraphLayer.from_dense_auto(adj, 3, seed=42, density_threshold=0.3)
    assert layer.is_sparse()


def test_from_dense_auto_dense():
    # 3x3 full → 100% density → should stay dense
    adj = np.ones((3, 3))
    layer = sc.StochasticGraphLayer.from_dense_auto(adj, 2, seed=42, density_threshold=0.3)
    assert not layer.is_sparse()


def test_sparse_matches_dense_output():
    adj = np.array([[0.0, 0.9, 0.0], [0.9, 0.0, 0.9], [0.0, 0.9, 0.0]])
    features = np.random.default_rng(7).random((3, 2))
    seed = 42

    dense_layer = sc.StochasticGraphLayer(adj, 2, seed=seed)
    sparse_layer = sc.StochasticGraphLayer.from_sparse(
        [0, 1, 3, 4], [1, 0, 2, 1], [0.9, 0.9, 0.9, 0.9], 3, 2, seed=seed
    )

    out_dense = np.array(dense_layer.forward(features))
    out_sparse = np.array(sparse_layer.forward(features))
    np.testing.assert_allclose(out_dense, out_sparse, atol=1e-12)
