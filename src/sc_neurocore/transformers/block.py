# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Spiking Transformer Block (S-Former)

from __future__ import annotations
from typing import Any
from dataclasses import dataclass
import numpy as np

from ..layers.attention import StochasticAttention
from ..layers.vectorized_layer import VectorizedSCLayer


@dataclass
class StochasticTransformerBlock:
    """
    Spiking Transformer Block (S-Former).
    Structure:
    Input -> Multi-Head Attention -> Add & Norm -> Feed Forward -> Add & Norm -> Output
    """

    d_model: int
    n_heads: int
    length: int = 1024

    def __post_init__(self) -> None:
        # We simplify Multi-Head to Single-Head for this demo
        self.attention = StochasticAttention(dim_k=self.d_model)

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
        """
        x: (d_model,) or (Sequence_Length, d_model). Returns same shape.
        """
        input_1d = x.ndim == 1
        attn_out = self.attention.forward(Q=x, K=x, V=x)

        # Match shapes for residual: attention may add a batch dim
        if input_1d and attn_out.ndim > 1:
            attn_out = attn_out.reshape(-1)[: x.shape[0]]

        res1 = np.clip(0.5 * x + 0.5 * attn_out, 0.0, 1.0)

        # Position-wise FFN: apply same weights to each token
        def _ffn(token: np.ndarray) -> np.ndarray:
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
