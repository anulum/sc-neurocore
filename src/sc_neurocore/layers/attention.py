# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations
from typing import Any, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class StochasticAttention:
    """
    Stochastic Computing Attention Block.

    Approximates: Output = Softmax(Q * K^T) * V
    """

    dim_k: int

    def forward(
        self, Q: np.ndarray[Any, Any], K: np.ndarray[Any, Any], V: np.ndarray[Any, Any]
    ) -> np.ndarray[Any, Any]:
        """
        Input:
            Q: (N, Dim_K) - Query Probabilities
            K: (M, Dim_K) - Key Probabilities
            V: (M, Dim_V) - Value Probabilities

        Returns:
            Output: (N, Dim_V)
        """
        # Ensure inputs are 2D (Seq/Batch, Dim)
        if Q.ndim == 1:
            Q = Q[None, :]
        if K.ndim == 1:
            K = K[None, :]
        if V.ndim == 1:
            V = V[None, :]

        # 1. Score Calculation (Matrix Multiplication)
        # Score = Q @ K.T -> (N, M)
        # In SC, this is parallel AND gates
        scores = np.dot(Q, K.T)

        # 2. Stochastic Softmax / Normalization
        # We normalize each row (each Query's attention over Keys)
        row_sums = np.sum(scores, axis=1, keepdims=True)
        # Avoid division by zero
        row_sums[row_sums == 0] = 1.0

        attn_weights = scores / row_sums

        # 3. Weighted Sum (V)
        # Out = attn_weights @ V -> (N, M) @ (M, Dim_V) -> (N, Dim_V)
        output = np.dot(attn_weights, V)

        return output  # type: ignore
