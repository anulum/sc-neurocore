# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Stochastic-computing synapse using bitstreams

from __future__ import annotations
from typing import Any, Optional
from dataclasses import dataclass
import math
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
        if not math.isfinite(self.w_min):
            raise ValueError("w_min must be finite")
        if not math.isfinite(self.w_max):
            raise ValueError("w_max must be finite")
        if self.w_min >= self.w_max:
            raise ValueError("w_min must be < w_max.")
        if type(self.length) is not int or self.length <= 0:
            raise ValueError("length must be a positive integer")
        self._validate_weight(self.w)

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
        self._validate_weight(new_w)
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
        if not isinstance(pre_bits, np.ndarray):
            raise ValueError("pre_bits must be a numpy array")
        if pre_bits.ndim != 1:
            raise ValueError("pre_bits must be a one-dimensional bitstream")
        if pre_bits.shape[0] != self.weight_bits.shape[0]:
            raise ValueError(
                f"Bitstream length mismatch: pre={pre_bits.shape[0]}, "
                f"weight={self.weight_bits.shape[0]}"
            )
        if not np.all((pre_bits == 0) | (pre_bits == 1)):
            raise ValueError("pre_bits must contain only binary values 0 or 1")

        # Logical AND implements multiplication in SC domain
        result: np.ndarray[Any, Any] = (pre_bits & self.weight_bits).astype(np.uint8)
        return result

    def effective_weight_probability(self) -> float:
        """
        Decode the weight bitstream's probability P(weight_bit=1).
        This is the effective unipolar probability representation.
        """
        return bitstream_to_probability(self.weight_bits)

    def _validate_weight(self, w: float) -> None:
        if not math.isfinite(w):
            raise ValueError("w must be finite")
        if not self.w_min <= w <= self.w_max:
            raise ValueError("w must be within [w_min, w_max]")
