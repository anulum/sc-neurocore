# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Event-Based Graph Convolution Layer

from typing import Any
import numpy as np


class StochasticGraphLayer:
    """
    Event-Based Graph Convolution Layer.
    Message Passing happens via Bitstreams.
    """

    def __init__(self, adj_matrix: np.ndarray[Any, Any], n_features: int):
        self.adj = adj_matrix  # (N, N)
        self.n_nodes = adj_matrix.shape[0]
        self.n_features = n_features
        # Weights: (Features, Features) - simple linear transform
        self.weights = np.random.uniform(0, 1, (n_features, n_features))

    def forward(self, node_features: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """
        node_features: (N, Features)
        """
        output = np.zeros_like(node_features)

        # 1. Message Passing (Aggregation)
        # For each node, sum neighbor features
        # In SC, this is MUX aggregation

        # Standard GCN: A * X * W

        # Aggregation:
        agg_features = np.dot(self.adj, node_features)
        # Normalize by degree? (Simplified)
        degrees = np.sum(self.adj, axis=1, keepdims=True)
        degrees[degrees == 0] = 1
        agg_features /= degrees

        # 2. Transformation (Linear)
        # Out = Agg * W
        output = np.dot(agg_features, self.weights)

        # 3. Non-linearity (Tanh/Sigmoid)
        activated: np.ndarray[Any, Any] = np.tanh(output)
        return activated
