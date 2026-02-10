"""Drop-in replacement for sc_neurocore.layers.StochasticAttention."""

from __future__ import annotations

import numpy as np

from sc_neurocore_engine.sc_neurocore_engine import StochasticAttention as _RustAttention


class StochasticAttention:
    """API-compatible with sc_neurocore.layers.StochasticAttention."""

    def __init__(self, dim_k: int):
        self.dim_k = dim_k
        self._engine = _RustAttention(dim_k)

    def forward(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
        Q = np.asarray(Q, dtype=np.float64)
        K = np.asarray(K, dtype=np.float64)
        V = np.asarray(V, dtype=np.float64)
        if Q.ndim == 1:
            Q = Q[None, :]
        if K.ndim == 1:
            K = K[None, :]
        if V.ndim == 1:
            V = V[None, :]
        result = self._engine.forward(Q, K, V)
        return np.asarray(result, dtype=np.float64)

    def forward_sc(
        self,
        Q: np.ndarray,
        K: np.ndarray,
        V: np.ndarray,
        length: int = 1024,
        seed: int = 44257,
    ) -> np.ndarray:
        Q = np.asarray(Q, dtype=np.float64)
        K = np.asarray(K, dtype=np.float64)
        V = np.asarray(V, dtype=np.float64)
        if Q.ndim == 1:
            Q = Q[None, :]
        if K.ndim == 1:
            K = K[None, :]
        if V.ndim == 1:
            V = V[None, :]
        result = self._engine.forward_sc(Q, K, V, int(length), int(seed))
        return np.asarray(result, dtype=np.float64)

    def forward_multihead(
        self,
        Q: np.ndarray,
        K: np.ndarray,
        V: np.ndarray,
        n_heads: int,
    ) -> np.ndarray:
        """Multi-head attention. Q/K/V columns split across heads."""
        Q = np.asarray(Q, dtype=np.float64)
        K = np.asarray(K, dtype=np.float64)
        V = np.asarray(V, dtype=np.float64)
        if Q.ndim == 1:
            Q = Q[None, :]
        if K.ndim == 1:
            K = K[None, :]
        if V.ndim == 1:
            V = V[None, :]
        result = self._engine.forward_multihead(Q, K, V, int(n_heads))
        return np.asarray(result, dtype=np.float64)
