# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Spike-native graph neural networks

"""Graph neural networks where messages are spike trains.

Nodes are spiking neuron populations. Messages = spike volleys.
Aggregation = coincidence detection (AND gates in SC domain).
Natural fit for stochastic computing FPGA deployment.

Reference: SGNNBench (2025) — 9 SGNN architectures benchmarked
"""

from __future__ import annotations

from typing import Any

from dataclasses import dataclass

import numpy as np


class SpikeGraphConv:
    """Spike-based graph convolution layer.

    Message passing: each node aggregates spike trains from neighbors,
    applies a learned weight transform via LIF integration.

    Parameters
    ----------
    in_features : int
        Input feature dimension per node.
    out_features : int
        Output feature dimension per node.
    threshold : float
    tau_mem : float
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        threshold: float = 1.0,
        tau_mem: float = 10.0,
        seed: int = 42,
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.threshold = threshold
        self.tau_mem = tau_mem

        rng = np.random.RandomState(seed)
        self.W = rng.randn(out_features, in_features) * np.sqrt(2.0 / in_features)
        self._v: np.ndarray[Any, Any] | None = None

    def forward(
        self,
        node_features: np.ndarray[Any, Any],
        adjacency: np.ndarray[Any, Any],
        T: int = 8,
    ) -> np.ndarray[Any, Any]:
        """Spike-based graph convolution.

        Parameters
        ----------
        node_features : ndarray of shape (N_nodes, in_features)
            Node features in [0, 1] (spike rates or encoded features).
        adjacency : ndarray of shape (N_nodes, N_nodes)
            Binary adjacency matrix (1 = edge, 0 = no edge).
        T : int
            Number of simulation timesteps.

        Returns
        -------
        ndarray of shape (N_nodes, out_features)
            Output spike counts per node per feature.
        """
        N = node_features.shape[0]
        rng = np.random.RandomState(42)

        # Aggregate neighbor features (message passing)
        degree = adjacency.sum(axis=1, keepdims=True)
        degree = np.clip(degree, 1, None)
        aggregated = (adjacency @ node_features) / degree

        # Project through weight matrix
        projected = aggregated @ self.W.T

        # LIF integration over T timesteps
        self._v = np.zeros((N, self.out_features))
        spike_counts = np.zeros((N, self.out_features))
        alpha = np.exp(-1.0 / self.tau_mem)

        for t in range(T):
            # Rate-code input: spike with probability proportional to projected value
            input_spikes = (rng.random(projected.shape) < np.clip(projected, 0, 1)).astype(
                np.float64
            )
            self._v = alpha * self._v + (1 - alpha) * input_spikes
            spikes = (self._v >= self.threshold).astype(np.float64)
            self._v -= spikes * self.threshold
            spike_counts += spikes

        return spike_counts


@dataclass
class SpikeGNNLayer:
    """Multi-layer spike GNN for graph classification/regression.

    Parameters
    ----------
    layer_dims : list of int
        [in_features, hidden1, ..., out_features]
    threshold : float
    T : int
        Simulation timesteps per layer.
    """

    layer_dims: list[int]
    threshold: float = 1.0
    T: int = 8

    def __post_init__(self) -> None:
        """Build the per-layer spiking graph-convolution stack."""
        self.convs = []
        for i in range(len(self.layer_dims) - 1):
            self.convs.append(
                SpikeGraphConv(
                    self.layer_dims[i],
                    self.layer_dims[i + 1],
                    threshold=self.threshold,
                    seed=42 + i,
                )
            )

    def forward(
        self, node_features: np.ndarray[Any, Any], adjacency: np.ndarray[Any, Any]
    ) -> np.ndarray[Any, Any]:
        """Forward pass through all layers.

        Parameters
        ----------
        node_features : ndarray of shape (N_nodes, in_features)
        adjacency : ndarray of shape (N_nodes, N_nodes)

        Returns
        -------
        ndarray of shape (N_nodes, out_features)
        """
        h = node_features
        for conv in self.convs:
            h = conv.forward(h, adjacency, T=self.T)
            # Normalize spike counts to [0, 1] for next layer
            max_val = h.max()
            if max_val > 0:  # pragma: no cover
                h = h / max_val
        return h

    def graph_classify(
        self, node_features: np.ndarray[Any, Any], adjacency: np.ndarray[Any, Any]
    ) -> int:
        """Classify a graph by global readout (sum pooling + argmax)."""
        node_out = self.forward(node_features, adjacency)
        graph_vec = node_out.sum(axis=0)
        return int(np.argmax(graph_vec))

    @property
    def n_layers(self) -> int:
        """Return the number of graph-convolution layers."""
        return len(self.convs)
