from __future__ import annotations
from typing import Any, Optional
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
        """
        Process a single time step, updating internal state and weight.
        Returns the post-synaptic current contribution (weighted bit).
        """
        # 1. Compute output contribution (standard SC behavior)
        # Get the current weight bit (we cycle through the stored bitstream or regenerate)
        # For simplicity in this step-wise function, we'll assume we access the weight_bits array
        # or generate a bit on the fly.
        # To match the 'apply' batch method, we might need a different approach.
        # BUT, STDP is inherently time-step dependent.

        # Let's assume we are generating weight bits dynamically for the update,
        # or we index into the existing static weight stream.
        # For true online learning, the weight probability itself changes.

        # Simplification: We generate a weight bit based on current w probability
        weight_bit = 1 if self._rng.random() < self.effective_weight_probability() else 0  # type: ignore
        output_bit = pre_bit & weight_bit

        # 2. STDP Update Logic

        # Push pre_bit to trace
        # (Shift buffer and add new bit)
        self._pre_trace = np.roll(self._pre_trace, 1)
        self._pre_trace[0] = pre_bit

        # LTP: If POST spikes (now), check recent PRE spikes
        if post_bit == 1:
            # Check if there was a pre spike in the window (excluding current step if desired)
            if np.any(self._pre_trace[1:]):
                if self._rng.random() < self.learning_rate:  # type: ignore
                    self._potentiate()  # type: ignore

        # LTD: If PRE spikes (now), check recent POST spikes
        # This is harder without a POST trace.
        # Alternative simple rule: If PRE=1 and POST=0 (within window), depress?
        # Standard STDP requires knowing if POST spiked *before* PRE.
        # We'll implement a simple "trace-based" approx:
        # If pre_bit=1, and we have a "post_trace" (not implemented yet), we depress.
        # For now, let's implement the LTP part primarily, or a simplified Hebbian rule:
        # If PRE=1 and POST=1 -> Potentiate
        # If PRE=1 and POST=0 -> Depress (Anti-Hebbian / Heterosynaptic)

        # Simplified Online Learning Rule:
        if pre_bit == 1 and post_bit == 1:
            if self._rng.random() < self.learning_rate:  # type: ignore
                self._potentiate()  # type: ignore
        elif pre_bit == 1 and post_bit == 0:
            if float(self._rng.random()) < (self.learning_rate * 0.5):  # type: ignore  # Depression is usually weaker  # type: ignore
                self._depress()  # type: ignore

        return output_bit

    def _potentiate(self):  # type: ignore
        """Increase weight probability."""
        new_w = min(self.w_max, self.w + self.learning_rate * (self.w_max - self.w_min))
        self.update_weight(new_w)

    def _depress(self):  # type: ignore
        """Decrease weight probability."""
        new_w = max(self.w_min, self.w - self.learning_rate * (self.w_max - self.w_min))
        self.update_weight(new_w)
