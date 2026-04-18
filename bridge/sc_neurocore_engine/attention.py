# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Drop-in replacement for

"""Drop-in replacement for sc_neurocore.layers.StochasticAttention."""

from __future__ import annotations

import numpy as np

from sc_neurocore_engine.sc_neurocore_engine import StochasticAttention as _RustAttention


def _to_2d(arr):
    a = np.asarray(arr, dtype=np.float64)
    return a[None, :] if a.ndim == 1 else a


class StochasticAttention:
    """API-compatible with sc_neurocore.layers.StochasticAttention."""

    def __init__(self, dim_k: int, temperature: float | None = None):
        self.dim_k = dim_k
        if temperature is not None:
            self._engine = _RustAttention(dim_k, temperature)
        else:
            self._engine = _RustAttention(dim_k)

    def forward(self, Q, K, V) -> np.ndarray:
        return np.asarray(self._engine.forward(_to_2d(Q), _to_2d(K), _to_2d(V)), dtype=np.float64)

    def forward_softmax(self, Q, K, V) -> np.ndarray:
        return np.asarray(
            self._engine.forward_softmax(_to_2d(Q), _to_2d(K), _to_2d(V)), dtype=np.float64
        )

    def forward_multihead_softmax(self, Q, K, V, n_heads: int) -> np.ndarray:
        return np.asarray(
            self._engine.forward_multihead_softmax(_to_2d(Q), _to_2d(K), _to_2d(V), int(n_heads)),
            dtype=np.float64,
        )

    def forward_sc(self, Q, K, V, length: int = 1024, seed: int = 44257) -> np.ndarray:
        return np.asarray(
            self._engine.forward_sc(_to_2d(Q), _to_2d(K), _to_2d(V), int(length), int(seed)),
            dtype=np.float64,
        )

    def forward_multihead(self, Q, K, V, n_heads: int) -> np.ndarray:
        return np.asarray(
            self._engine.forward_multihead(_to_2d(Q), _to_2d(K), _to_2d(V), int(n_heads)),
            dtype=np.float64,
        )
