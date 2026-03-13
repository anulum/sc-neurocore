# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations
from typing import Any
from dataclasses import dataclass, field
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
        # Buffer to store recent pre-synaptic bits
        self._pre_trace = np.zeros(self.window_size, dtype=np.uint8)

    def process_step(self, pre_bit: int, post_bit: int) -> int:
        """Process one timestep: compute output, update trace, apply STDP."""
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

    def _potentiate(self):
        new_w = min(self.w_max, self.w + self.learning_rate * (self.w_max - self.w_min))
        self.update_weight(new_w)

    def _depress(self):
        new_w = max(self.w_min, self.w - self.learning_rate * (self.w_max - self.w_min))
        self.update_weight(new_w)
