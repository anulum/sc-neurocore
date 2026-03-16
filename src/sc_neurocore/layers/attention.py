# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

from __future__ import annotations
from typing import Any
from dataclasses import dataclass
import numpy as np


@dataclass
class StochasticAttention:
    """
    Stochastic Computing Attention Block.

    Two modes:

    - ``forward()`` — row-sum normalised (SC-native, no exp). Matches Rust engine ``forward()``.
    - ``forward_softmax()`` — proper softmax with temperature scaling.

    Example
    -------
    >>> Q = np.random.default_rng(0).uniform(0, 1, (4, 8))
    >>> K = np.random.default_rng(1).uniform(0, 1, (6, 8))
    >>> V = np.random.default_rng(2).uniform(0, 1, (6, 5))
    >>> attn = StochasticAttention(dim_k=8)
    >>> attn.forward(Q, K, V).shape
    (4, 5)
    >>> attn.forward_softmax(Q, K, V).shape
    (4, 5)
    """

    dim_k: int
    temperature: float = 1.0

    def _ensure_2d(
        self,
        Q: np.ndarray[Any, Any],
        K: np.ndarray[Any, Any],
        V: np.ndarray[Any, Any],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if Q.ndim == 1:
            Q = Q[None, :]
        if K.ndim == 1:
            K = K[None, :]
        if V.ndim == 1:
            V = V[None, :]
        return Q, K, V

    def forward(
        self, Q: np.ndarray[Any, Any], K: np.ndarray[Any, Any], V: np.ndarray[Any, Any]
    ) -> np.ndarray[Any, Any]:
        """
        Row-sum normalised attention (SC-native, no exp).

        Parameters
        ----------
        Q : (N, dim_k)
        K : (M, dim_k)
        V : (M, dim_v)

        Returns
        -------
        (N, dim_v)
        """
        Q, K, V = self._ensure_2d(Q, K, V)
        scores = np.dot(Q, K.T)
        row_sums = np.sum(scores, axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        attn_weights = scores / row_sums
        return np.dot(attn_weights, V)

    def forward_softmax(
        self, Q: np.ndarray[Any, Any], K: np.ndarray[Any, Any], V: np.ndarray[Any, Any]
    ) -> np.ndarray[Any, Any]:
        """
        Proper softmax attention with temperature scaling.

        softmax(Q @ K^T / temperature) @ V

        Numerically stable via max-subtraction before exp.

        Parameters
        ----------
        Q : (N, dim_k)
        K : (M, dim_k)
        V : (M, dim_v)

        Returns
        -------
        (N, dim_v)
        """
        Q, K, V = self._ensure_2d(Q, K, V)
        scores = np.dot(Q, K.T) / self.temperature
        scores -= scores.max(axis=1, keepdims=True)
        exp_scores = np.exp(scores)
        attn_weights = exp_scores / exp_scores.sum(axis=1, keepdims=True)
        return np.dot(attn_weights, V)
