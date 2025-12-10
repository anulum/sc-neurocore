from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any

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
    seed: int | None = None

    def __post_init__(self) -> None:
        self._rng = RNG(self.seed)
        self.v = self.v_rest  # membrane potential
        self.reset_state()

    def step(self, input_current: float) -> int:
        # Membrane leak term
        dv_leak = -(self.v - self.v_rest) * (self.dt / self.tau_mem)

        # Input term (simple Ohm's law; you can absorb R into current)
        dv_input = self.resistance * input_current * self.dt

        # Noise term
        dv_noise = 0.0
        if self.noise_std > 0.0:
            dv_noise = float(self._rng.normal(0.0, self.noise_std))

        # Update membrane potential
        self.v += dv_leak + dv_input + dv_noise

        # Check for spike
        if self.v >= self.v_threshold:
            spike = 1
            self.v = self.v_reset
        else:
            spike = 0
        return spike

    def reset_state(self) -> None:
        self.v = self.v_rest

    def get_state(self) -> Dict[str, Any]:
        return {"v": float(self.v)}
