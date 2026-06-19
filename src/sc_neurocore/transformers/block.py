# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Spiking Transformer Block (S-Former)

"""Spiking transformer block (S-Former) with stochastic multi-head attention."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from ..layers.attention import StochasticAttention
from ..layers.vectorized_layer import VectorizedSCLayer


class _AttentionHead(Protocol):
    dim_k: int

    def forward(
        self,
        Q: np.ndarray[Any, Any],
        K: np.ndarray[Any, Any],
        V: np.ndarray[Any, Any],
    ) -> np.ndarray[Any, Any]: ...


@dataclass
class StochasticTransformerBlock:
    """Spiking transformer block (S-Former).

    Structure: input -> multi-head attention -> add & norm -> feed forward
    -> add & norm -> output.
    """

    d_model: int
    n_heads: int
    length: int = 1024

    def __post_init__(self) -> None:
        """Validate the block dimensions and build the attention heads and FFN."""
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.n_heads <= 0:
            raise ValueError("n_heads must be positive")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.length <= 0:
            raise ValueError("length must be positive")

        self.head_dim = self.d_model // self.n_heads
        self.attention_heads: list[_AttentionHead] = [
            StochasticAttention(dim_k=self.head_dim) for _ in range(self.n_heads)
        ]

        # Feed Forward Network (FFN)
        # 2-layer MLP: d_model -> 4*d_model -> d_model
        # We use our Vectorized Layer for efficiency
        self.ffn_1 = VectorizedSCLayer(
            n_inputs=self.d_model, n_neurons=4 * self.d_model, length=self.length
        )
        self.ffn_2 = VectorizedSCLayer(
            n_inputs=4 * self.d_model, n_neurons=self.d_model, length=self.length
        )

    def forward(self, x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Run the block on input of shape (d_model,) or (seq_len, d_model)."""
        x = np.asarray(x, dtype=np.float64)
        if x.ndim not in (1, 2):
            raise ValueError("x must be a one- or two-dimensional array")
        if x.shape[-1] != self.d_model:
            raise ValueError(f"x must have trailing dimension d_model={self.d_model}")
        if not np.all(np.isfinite(x)):
            raise ValueError("x must contain only finite values")

        input_1d = x.ndim == 1
        x_2d = x[None, :] if input_1d else x
        attn_out = self._multi_head_attention(x_2d)

        # Match shapes for residual: attention may add a batch dim
        if input_1d:
            attn_out = attn_out.reshape(-1)[: x.shape[0]]

        res1 = np.clip(0.5 * x + 0.5 * attn_out, 0.0, 1.0)

        # Position-wise FFN: apply same weights to each token
        def _ffn(token: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
            vals = token.tolist() if hasattr(token, "tolist") else token
            h = np.clip(self.ffn_1.forward(vals), 0.0, 1.0)  # type: ignore[arg-type]
            return self.ffn_2.forward(h.tolist() if hasattr(h, "tolist") else h)  # type: ignore[arg-type]

        if res1.ndim > 1:
            ff_out = np.zeros_like(res1)
            for t in range(res1.shape[0]):
                ff_out[t] = _ffn(res1[t])
        else:
            ff_out = _ffn(res1)

        return 0.5 * res1 + 0.5 * ff_out

    def _multi_head_attention(self, x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        x_2d = np.asarray(x, dtype=np.float64)
        if x_2d.ndim != 2 or x_2d.shape[1] != self.d_model:
            raise ValueError(f"x must have shape (sequence_length, {self.d_model})")

        head_outputs = []
        for head_idx, head in enumerate(self.attention_heads):
            start = head_idx * self.head_dim
            stop = start + self.head_dim
            head_x = x_2d[:, start:stop]
            head_outputs.append(head.forward(Q=head_x, K=head_x, V=head_x))

        return np.concatenate(head_outputs, axis=1)
