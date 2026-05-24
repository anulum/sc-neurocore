# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Stochastic synapse with spike-timing-dependent plasticity

from __future__ import annotations
from typing import Any
from dataclasses import dataclass, field
import math
import numpy as np

from .sc_synapse import BitstreamSynapse
from ..constants import STDP_LEARNING_RATE, STDP_WINDOW_SIZE, STDP_LTD_RATIO


@dataclass
class StochasticSTDPSynapse(BitstreamSynapse):
    """
    Stochastic synapse with spike-timing-dependent plasticity.

    LTP on pre→post coincidence, LTD on pre-without-post.
    Asymmetry ratio from Bi & Poo, J. Neurosci. 18(24), 1998.

    Example
    -------
    >>> syn = StochasticSTDPSynapse(w_min=0.0, w_max=1.0, w=0.5, length=64)
    >>> for _ in range(100):
    ...     syn.process_step(pre_bit=1, post_bit=1)  # correlated activity → LTP
    >>> syn.w >= 0.5  # weight increased or stayed
    True
    """

    learning_rate: float = STDP_LEARNING_RATE
    window_size: int = STDP_WINDOW_SIZE
    ltd_ratio: float = STDP_LTD_RATIO

    _pre_trace: np.ndarray[Any, Any] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        if not math.isfinite(self.learning_rate) or not 0.0 <= self.learning_rate <= 1.0:
            raise ValueError("learning_rate must be finite and within [0, 1]")
        if type(self.window_size) is not int or self.window_size <= 0:
            raise ValueError("window_size must be a positive integer")
        if not math.isfinite(self.ltd_ratio) or self.ltd_ratio < 0.0:
            raise ValueError("ltd_ratio must be finite and non-negative")

        # Buffer to store recent pre-synaptic bits
        self._pre_trace = np.zeros(self.window_size, dtype=np.uint8)

    def process_step(self, pre_bit: int, post_bit: int) -> int:
        """Process one timestep: compute output, update trace, apply STDP."""
        self._validate_bit("pre_bit", pre_bit)
        self._validate_bit("post_bit", post_bit)

        weight_bit = 1 if self._rng.random() < self.effective_weight_probability() else 0
        output_bit = pre_bit & weight_bit

        self._pre_trace = np.roll(self._pre_trace, 1)
        self._pre_trace[0] = pre_bit

        # Trace-based STDP: post spike + recent pre activity → LTP.
        # Pre spike without post → LTD. Mutually exclusive per timestep.
        if post_bit == 1 and np.any(self._pre_trace[1:]):
            if self._rng.random() < self.learning_rate:
                self._potentiate()
        elif pre_bit == 1 and post_bit == 0:
            if self._rng.random() < self.learning_rate * self.ltd_ratio:
                self._depress()

        return output_bit

    def _potentiate(self) -> None:
        new_w = min(self.w_max, self.w + self.learning_rate * (self.w_max - self.w_min))
        self.update_weight(new_w)

    def _depress(self) -> None:
        new_w = max(self.w_min, self.w - self.learning_rate * (self.w_max - self.w_min))
        self.update_weight(new_w)

    @staticmethod
    def _validate_bit(name: str, value: int) -> None:
        if type(value) is not int or value not in (0, 1):
            raise ValueError(f"{name} must be an integer bit, 0 or 1")
