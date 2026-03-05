# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations
from typing import Any, Optional
from dataclasses import dataclass
from typing import Optional
import numpy as np

from ..utils.bitstreams import (
    BitstreamEncoder,
    bitstream_to_probability,
)
from ..utils.rng import RNG


@dataclass
class BitstreamSynapse:
    """
    Stochastic-computing synapse using bitstreams.

    Each synapse has a weight w in [w_min, w_max].
    For 'pure' SC mode, we encode w as a bitstream and AND it with the
    pre-synaptic bitstream:

        out_bit[t] = pre_bit[t] & weight_bit[t]

    In expectation:
        P(out=1) ≈ P(pre=1) * P(weight=1)
    which corresponds to multiplication of the underlying probabilities.

    For now we implement:
    - encode_weight() -> weight_bitstream
    - apply(pre_bits) -> post_bits
    """

    w_min: float
    w_max: float
    length: int = 256
    w: float = 0.5
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if self.w_min >= self.w_max:
            raise ValueError("w_min must be < w_max.")
        self._rng = RNG(self.seed)
        self._weight_encoder = BitstreamEncoder(
            x_min=self.w_min,
            x_max=self.w_max,
            length=self.length,
            seed=self.seed,
        )
        self.weight_bits = self.encode_weight(self.w)

    def encode_weight(self, w: float) -> np.ndarray[Any, Any]:
        """
        Encode scalar weight w into a unipolar bitstream.
        """
        return self._weight_encoder.encode(w)

    def update_weight(self, new_w: float) -> None:
        """
        Change synaptic weight and recompute its bitstream.
        """
        self.w = new_w
        self.weight_bits = self.encode_weight(new_w)

    def apply(self, pre_bits: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """
        Apply synapse to a pre-synaptic bitstream.

        Parameters
        ----------
        pre_bits : np.ndarray
            Bitstream of shape (length,) with values {0,1}.

        Returns
        -------
        np.ndarray
            Post-synaptic bitstream of shape (length,).
        """
        if pre_bits.shape[0] != self.weight_bits.shape[0]:
            raise ValueError(
                f"Bitstream length mismatch: pre={pre_bits.shape[0]}, "
                f"weight={self.weight_bits.shape[0]}"
            )
        # Logical AND implements multiplication in SC domain
        return (pre_bits & self.weight_bits).astype(np.uint8)  # type: ignore

    def effective_weight_probability(self) -> float:
        """
        Decode the weight bitstream's probability P(weight_bit=1).
        This is the effective unipolar probability representation.
        """
        return bitstream_to_probability(self.weight_bits)
