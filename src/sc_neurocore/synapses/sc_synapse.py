# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

from __future__ import annotations
from typing import Any, Optional
from dataclasses import dataclass
import numpy as np

from ..utils.bitstreams import (
    BitstreamEncoder,
    bitstream_to_probability,
)
from ..utils.rng import RNG
from ..constants import SYNAPSE_DEFAULT_LENGTH, SYNAPSE_DEFAULT_WEIGHT


@dataclass
class BitstreamSynapse:
    """
    Stochastic-computing synapse using bitstreams.

    Each synapse has a weight w in [w_min, w_max].
    SC multiplication via bitwise AND: P(out=1) ~ P(pre=1) * P(w=1).

    Example
    -------
    >>> import numpy as np
    >>> syn = BitstreamSynapse(w_min=0.0, w_max=1.0, w=0.5, length=1024, seed=42)
    >>> pre = np.ones(1024, dtype=np.uint8)  # all-ones input
    >>> post = syn.apply(pre)
    >>> abs(post.mean() - 0.5) < 0.1  # output ~50% ones
    True
    """

    w_min: float
    w_max: float
    length: int = SYNAPSE_DEFAULT_LENGTH
    w: float = SYNAPSE_DEFAULT_WEIGHT
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
        return (pre_bits & self.weight_bits).astype(np.uint8)

    def effective_weight_probability(self) -> float:
        """
        Decode the weight bitstream's probability P(weight_bit=1).
        This is the effective unipolar probability representation.
        """
        return bitstream_to_probability(self.weight_bits)
