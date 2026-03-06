# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations
from typing import Any
from dataclasses import dataclass
from typing import Dict

import numpy as np

from .base import BaseNeuron
from ..utils.rng import RNG


@dataclass
class StochasticLIFNeuron(BaseNeuron):
    """
    Discrete-time noisy leaky integrate-and-fire neuron.

    dv/dt = -(v - v_rest) / tau_mem + R * I + noise

    We work in simple units:
    - dt: time step
    - tau_mem: membrane time constant
    - v_threshold: firing threshold
    - v_reset: reset potential
    - noise_std: std dev of Gaussian noise added each step
    """

    v_rest: float = 0.0
    v_reset: float = 0.0
    v_threshold: float = 1.0
    tau_mem: float = 20.0
    dt: float = 1.0
    noise_std: float = 0.0
    resistance: float = 1.0
    refractory_period: int = 0
    seed: int | None = None
    entropy_source: Any | None = None  # Optional external entropy (e.g. Quantum)

    def __post_init__(self) -> None:
        self._rng = RNG(self.seed)
        self.v = self.v_rest  # membrane potential
        self.refractory_counter = 0
        self.reset_state()

    def step(self, input_current: float) -> int:
        if self.refractory_counter > 0:
            self.refractory_counter -= 1
            self.v = self.v_rest
            return 0

        # Membrane leak term
        dv_leak = -(self.v - self.v_rest) * (self.dt / self.tau_mem)

        # Input term (simple Ohm's law; you can absorb R into current)
        dv_input = self.resistance * input_current * self.dt

        # Noise term
        dv_noise = 0.0
        if self.noise_std > 0.0:
            if self.entropy_source is not None:
                # Use external (Quantum) source
                dv_noise = float(self.entropy_source.sample_normal(0.0, self.noise_std))
            else:
                # Use internal (Pseudo-random) source
                dv_noise = float(self._rng.normal(0.0, self.noise_std))

        # Update membrane potential
        self.v += dv_leak + dv_input + dv_noise

        # Check for spike
        if self.v >= self.v_threshold:
            spike = 1
            self.v = self.v_reset
            self.refractory_counter = self.refractory_period
        else:
            spike = 0
        return spike

    def reset_state(self) -> None:
        self.v = self.v_rest
        self.refractory_counter = 0

    def get_state(self) -> Dict[str, Any]:
        return {"v": float(self.v), "refractory": self.refractory_counter}

    def process_bitstream(
        self, input_bits: np.ndarray[Any, Any], input_scale: float = 1.0
    ) -> np.ndarray[Any, Any]:
        """
        Process a bitstream (array of 0s and 1s) as input current.
        Returns an array of spikes (0s and 1s).

        input_scale: scaling factor to convert bit (0/1) to current amplitude.
        """
        spikes = np.zeros_like(input_bits, dtype=np.uint8)
        for i, bit in enumerate(input_bits):
            # Treat bit as current pulse of amplitude 'input_scale'
            current = bit * input_scale
            spikes[i] = self.step(current)
        return spikes
