"""Drop-in replacement for sc_neurocore.graphs.StochasticGraphLayer."""

from __future__ import annotations

import numpy as np

from sc_neurocore_engine.sc_neurocore_engine import StochasticGraphLayer as _RustGraphLayer


class StochasticGraphLayer:
    """API-compatible with sc_neurocore.graphs.StochasticGraphLayer."""

    def __init__(self, adj_matrix: np.ndarray, n_features: int, seed: int = 42):
        adj = np.asarray(adj_matrix, dtype=np.float64)
        if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
            raise ValueError(f"adj_matrix must be square 2-D, got shape {adj.shape}")
        self.n_nodes = int(adj.shape[0])
        self.n_features = int(n_features)
        self._engine = _RustGraphLayer(adj, self.n_features, int(seed))
        self.weights = np.array(self._engine.get_weights(), dtype=np.float64).reshape(
            self.n_features, self.n_features
        )

    def forward(self, node_features: np.ndarray) -> np.ndarray:
        X = np.asarray(node_features, dtype=np.float64)
        result = self._engine.forward(X)
        return np.asarray(result, dtype=np.float64).reshape(self.n_nodes, self.n_features)

    def forward_sc(
        self,
        node_features: np.ndarray,
        length: int = 1024,
        seed: int = 44257,
    ) -> np.ndarray:
        """SC-mode forward pass using bitstream AND+popcount."""
        X = np.asarray(node_features, dtype=np.float64)
        result = self._engine.forward_sc(X, int(length), int(seed))
        return np.asarray(result, dtype=np.float64).reshape(self.n_nodes, self.n_features)
