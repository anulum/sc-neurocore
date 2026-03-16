# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Drop-in replacement for sc_neurocore.graphs.StochasticGra...

"""Drop-in replacement for sc_neurocore.graphs.StochasticGraphLayer."""

from __future__ import annotations

import numpy as np

from sc_neurocore_engine.sc_neurocore_engine import StochasticGraphLayer as _RustGraphLayer


class StochasticGraphLayer:
    """API-compatible with sc_neurocore.graphs.StochasticGraphLayer."""

    def __init__(self, adj_matrix, n_features: int, seed: int = 42):
        adj = np.asarray(adj_matrix, dtype=np.float64)
        if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
            raise ValueError(f"adj_matrix must be square 2-D, got shape {adj.shape}")
        self.n_nodes = int(adj.shape[0])
        self.n_features = int(n_features)
        self._engine = _RustGraphLayer(adj, self.n_features, int(seed))
        self.weights = np.array(self._engine.get_weights(), dtype=np.float64).reshape(
            self.n_features, self.n_features
        )

    @classmethod
    def from_sparse(
        cls,
        row_offsets: list[int],
        col_indices: list[int],
        values: list[float],
        n_nodes: int,
        n_features: int,
        seed: int = 42,
    ) -> StochasticGraphLayer:
        obj = object.__new__(cls)
        obj.n_nodes = n_nodes
        obj.n_features = n_features
        obj._engine = _RustGraphLayer.from_sparse(
            row_offsets, col_indices, values, n_nodes, n_features, seed=seed
        )
        obj.weights = np.array(obj._engine.get_weights(), dtype=np.float64).reshape(
            n_features, n_features
        )
        return obj

    @classmethod
    def from_dense_auto(
        cls,
        adj_matrix,
        n_features: int,
        seed: int = 42,
        density_threshold: float = 0.3,
    ) -> StochasticGraphLayer:
        adj = np.asarray(adj_matrix, dtype=np.float64)
        n_nodes = int(adj.shape[0])
        obj = object.__new__(cls)
        obj.n_nodes = n_nodes
        obj.n_features = n_features
        obj._engine = _RustGraphLayer.from_dense_auto(
            adj, n_features, seed=seed, density_threshold=density_threshold
        )
        obj.weights = np.array(obj._engine.get_weights(), dtype=np.float64).reshape(
            n_features, n_features
        )
        return obj

    def is_sparse(self) -> bool:
        return self._engine.is_sparse()

    def forward(self, node_features) -> np.ndarray:
        X = np.asarray(node_features, dtype=np.float64)
        result = self._engine.forward(X)
        return np.asarray(result, dtype=np.float64).reshape(self.n_nodes, self.n_features)

    def forward_sc(self, node_features, length: int = 1024, seed: int = 44257) -> np.ndarray:
        X = np.asarray(node_features, dtype=np.float64)
        result = self._engine.forward_sc(X, int(length), int(seed))
        return np.asarray(result, dtype=np.float64).reshape(self.n_nodes, self.n_features)
