# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations
from typing import Any
from dataclasses import dataclass, field
import numpy as np

from .sc_synapse import BitstreamSynapse


@dataclass
class StochasticSTDPSynapse(BitstreamSynapse):
    """
    Stochastic Synapse with Spike-Timing-Dependent Plasticity (STDP).

    Implements a simplified stochastic STDP rule:
    - If PRE spikes and POST spikes shortly after -> Potentiation (LTP)
    - If POST spikes and PRE spikes shortly after -> Depression (LTD)

    Instead of full trace tracking, we use a probabilistic update based on
    coincidence windows.
    """

    learning_rate: float = 0.01  # Probability of weight update on event
    window_size: int = 5  # Time window for correlation in bits

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
            if self._rng.random() < self.learning_rate * 0.5:
                self._depress()

        return output_bit

    def _potentiate(self):
        new_w = min(self.w_max, self.w + self.learning_rate * (self.w_max - self.w_min))
        self.update_weight(new_w)

    def _depress(self):
        new_w = max(self.w_min, self.w - self.learning_rate * (self.w_max - self.w_min))
        self.update_weight(new_w)
