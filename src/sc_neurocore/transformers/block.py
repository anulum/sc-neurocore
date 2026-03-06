# SPDX-License-Identifier: AGPL-3.0-or-later
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

    def __post_init__(self):  # type: ignore
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
        x: (Sequence_Length, d_model) - Probabilities [0,1]
        """
        # 1. Self-Attention
        # Q, K, V are all projections of x. For simplicity, we assume Identity projection.
        attn_out = self.attention.forward(Q=x, K=x, V=x)

        # 2. Add & Norm (Residual 1)
        # SC Addition: (A + B) / 2 using MUX logic probability
        res1 = 0.5 * x + 0.5 * attn_out

        # 3. Feed Forward
        # Vectorized layer returns 1D array of size n_neurons. We need to reshape?
        # Our VectorizedSCLayer is "Dense", it fully connects all inputs to all outputs.
        # But Transformer FFN applies to each position independently (Position-wise).
        # We'll simulate position-wise by looping or reshaping.

        # Simplification: Apply to whole sequence as one vector (Global MLP)
        # Reshape to match layer expectation if needed.
        # Our VectorizedLayer takes flat input.

        # Let's map properly: Input (Seq, D) -> Flat (Seq*D)
        # But weights are (4D, Seq*D)?? No, standard FFN is shared weights.
        # We need a shared-weight layer application.

        # For this demo, let's assume Sequence Length = 1 (Context vector)
        # x is (1, d_model) or just (d_model,)
        if x.ndim > 1:
            x_flat = x[0]  # Take first token
        else:
            x_flat = x

        ff1_res = self.ffn_1.forward(x_flat.tolist() if hasattr(x_flat, "tolist") else x_flat)
        ff2_res = self.ffn_2.forward(ff1_res.tolist() if hasattr(ff1_res, "tolist") else ff1_res)  # type: ignore

        # 4. Add & Norm (Residual 2)
        final_out = 0.5 * res1.flatten() + 0.5 * ff2_res

        return final_out
