# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Drop-in replacement for

"""Drop-in replacement for sc_neurocore.graphs.StochasticGraphLayer."""

from __future__ import annotations

from typing import TYPE_CHECKING as _TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from sc_neurocore_engine.sc_neurocore_engine import StochasticGraphLayer as _RustGraphLayer

if _TYPE_CHECKING:
    from ._binding_types import StochasticGraphLayer as _GraphType


class StochasticGraphLayer:
    """API-compatible with sc_neurocore.graphs.StochasticGraphLayer."""

    def __init__(self, adj_matrix: npt.ArrayLike, n_features: int, seed: int = 42) -> None:
        """Construct a native graph layer from a square adjacency matrix.

        Parameters
        ----------
        adj_matrix : array_like
            Square node adjacency matrix.
        n_features : int
            Number of features at each node.
        seed : int, optional
            Native weight-initialisation seed.
        """
        adj = np.asarray(adj_matrix, dtype=np.float64)
        if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
            raise ValueError(f"adj_matrix must be square 2-D, got shape {adj.shape}")
        self.n_nodes = int(adj.shape[0])
        self.n_features = int(n_features)
        self._engine: _GraphType = _RustGraphLayer(adj, self.n_features, int(seed))
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
        """Construct a native graph layer from CSR adjacency arrays.

        Parameters
        ----------
        row_offsets, col_indices : list of int
            CSR offsets and column indices.
        values : list of float
            CSR edge weights.
        n_nodes, n_features : int
            Node count and number of features per node.
        seed : int, optional
            Native weight-initialisation seed.

        Returns
        -------
        StochasticGraphLayer
            Sparse graph layer with native weights.
        """
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
        adj_matrix: npt.ArrayLike,
        n_features: int,
        seed: int = 42,
        density_threshold: float = 0.3,
    ) -> StochasticGraphLayer:
        """Select the native adjacency representation using the requested density threshold.

        Parameters
        ----------
        adj_matrix : array_like
            Square node adjacency matrix.
        n_features : int
            Number of features at each node.
        seed : int, optional
            Native weight-initialisation seed.
        density_threshold : float, optional
            Density below which native storage uses CSR.

        Returns
        -------
        StochasticGraphLayer
            Graph layer with the native selected representation.
        """
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
        """Report whether the native graph uses CSR adjacency storage.

        Returns
        -------
        bool
            True for the sparse native representation.
        """
        return self._engine.is_sparse()

    def forward(self, node_features: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Aggregate node features using the native graph kernel.

        Parameters
        ----------
        node_features : array_like
            Feature matrix shaped as nodes by features.

        Returns
        -------
        ndarray of float64
            Aggregated features with the same nodes-by-features shape.
        """
        X = np.asarray(node_features, dtype=np.float64)
        result = self._engine.forward(X)
        return np.asarray(result, dtype=np.float64).reshape(self.n_nodes, self.n_features)

    def forward_sc(
        self, node_features: npt.ArrayLike, length: int = 1024, seed: int = 44257
    ) -> npt.NDArray[np.float64]:
        """Aggregate features through native stochastic bitstream multiplication.

        Parameters
        ----------
        node_features : array_like
            Feature matrix shaped as nodes by features.
        length : int, optional
            Positive stochastic bitstream length.
        seed : int, optional
            Replayable stochastic encoding seed.

        Returns
        -------
        ndarray of float64
            Aggregated features shaped as nodes by features.
        """
        X = np.asarray(node_features, dtype=np.float64)
        result = self._engine.forward_sc(X, int(length), int(seed))
        return np.asarray(result, dtype=np.float64).reshape(self.n_nodes, self.n_features)
