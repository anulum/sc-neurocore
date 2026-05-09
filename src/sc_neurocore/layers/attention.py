# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Stochastic Computing Attention Block

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from ..utils.bitstreams import (
    generate_bernoulli_bitstream,
    generate_sobol_bitstream,
)

SCMode = Literal["unipolar"]


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
    sc_mode: SCMode = "unipolar"

    def __post_init__(self) -> None:
        if self.dim_k <= 0:
            raise ValueError("dim_k must be positive")
        if self.temperature <= 0.0 or not np.isfinite(self.temperature):
            raise ValueError("temperature must be finite and positive")
        if self.sc_mode != "unipolar":
            raise ValueError("StochasticAttention currently supports only sc_mode='unipolar'")

    def _ensure_2d(
        self,
        Q: np.ndarray[Any, Any],
        K: np.ndarray[Any, Any],
        V: np.ndarray[Any, Any],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        Q = np.asarray(Q, dtype=np.float64)
        K = np.asarray(K, dtype=np.float64)
        V = np.asarray(V, dtype=np.float64)
        if Q.ndim == 1:
            Q = Q[None, :]
        if K.ndim == 1:
            K = K[None, :]
        if V.ndim == 1:
            V = V[None, :]
        if Q.ndim != 2 or K.ndim != 2 or V.ndim != 2:
            raise ValueError("Q, K, and V must be one- or two-dimensional arrays")
        if Q.shape[1] != self.dim_k:
            raise ValueError(f"Q must have dim_k={self.dim_k} columns")
        if K.shape[1] != self.dim_k:
            raise ValueError(f"K must have dim_k={self.dim_k} columns")
        if V.shape[0] != K.shape[0]:
            raise ValueError("V must have the same number of rows as K")
        if not (np.all(np.isfinite(Q)) and np.all(np.isfinite(K)) and np.all(np.isfinite(V))):
            raise ValueError("Q, K, and V must contain only finite values")
        return Q, K, V

    def _validate_unipolar_bitstream_inputs(
        self,
        Q: NDArray[np.float64],
        K: NDArray[np.float64],
        V: NDArray[np.float64],
        length: int,
    ) -> None:
        if length <= 0:
            raise ValueError("length must be positive")
        for name, array in (("Q", Q), ("K", K), ("V", V)):
            if np.any((array < 0.0) | (array > 1.0)):
                raise ValueError(
                    f"{name} values must be in [0, 1] for unipolar bitstream attention"
                )

    def forward(
        self, Q: np.ndarray[Any, Any], K: np.ndarray[Any, Any], V: np.ndarray[Any, Any]
    ) -> NDArray[np.float64]:
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
        return np.asarray(np.dot(attn_weights, V), dtype=np.float64)

    def forward_softmax(
        self, Q: np.ndarray[Any, Any], K: np.ndarray[Any, Any], V: np.ndarray[Any, Any]
    ) -> NDArray[np.float64]:
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
        return np.asarray(np.dot(attn_weights, V), dtype=np.float64)

    def forward_bitstream(
        self,
        Q: np.ndarray[Any, Any],
        K: np.ndarray[Any, Any],
        V: np.ndarray[Any, Any],
        length: int = 1024,
        use_sobol: bool = False,
    ) -> NDArray[np.float64]:
        """SC-native attention via bitstream AND gates.

        Each element is encoded as a bitstream, inner products computed
        via AND (bit-level multiply), results decoded by popcount.

        When use_sobol=True, Sobol low-discrepancy sequences replace
        Bernoulli random streams, reducing variance from O(1/√L) to O(1/L).

        Parameters
        ----------
        Q : (N, dim_k) — query probabilities in [0, 1]
        K : (M, dim_k) — key probabilities in [0, 1]
        V : (M, dim_v) — value probabilities in [0, 1]
        length : int — bitstream length
        use_sobol : bool — use Sobol sequences for variance reduction

        Returns
        -------
        (N, dim_v) — attention output probabilities
        """
        Q, K, V = self._ensure_2d(Q, K, V)
        self._validate_unipolar_bitstream_inputs(Q, K, V, length)
        N, dk = Q.shape
        M, dv = V.shape

        gen = generate_sobol_bitstream if use_sobol else generate_bernoulli_bitstream

        # Encode Q, K as bitstreams
        Q_bits = np.array(
            [[gen(float(Q[i, d]), length) for d in range(dk)] for i in range(N)]
        )  # (N, dk, L)
        K_bits = np.array(
            [[gen(float(K[j, d]), length) for d in range(dk)] for j in range(M)]
        )  # (M, dk, L)

        # Compute attention scores via AND (SC multiply) + popcount
        scores = np.zeros((N, M))
        for i in range(N):
            for j in range(M):
                # Inner product: sum of AND across dim_k
                and_sum = 0.0
                for d in range(dk):
                    and_result = np.bitwise_and(Q_bits[i, d], K_bits[j, d])
                    and_sum += np.sum(and_result)
                scores[i, j] = and_sum / (dk * length)

        # Row-sum normalization (SC-native, no exp)
        row_sums = scores.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        attn_weights = scores / row_sums

        # Weighted sum over V
        return np.asarray(np.dot(attn_weights, V), dtype=np.float64)
