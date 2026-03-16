# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Discrete-time noisy leaky integrate-and-fire neuron

from __future__ import annotations
from typing import Any
from dataclasses import dataclass
from typing import Dict

import numpy as np

from .base import BaseNeuron
from ..utils.rng import RNG
from ..constants import (
    LIF_V_REST,
    LIF_V_RESET,
    LIF_V_THRESHOLD,
    LIF_TAU_MEM,
    LIF_DT,
    LIF_NOISE_STD,
    LIF_RESISTANCE,
    LIF_REFRACTORY_PERIOD,
)


@dataclass
class StochasticLIFNeuron(BaseNeuron):
    """
    Discrete-time noisy leaky integrate-and-fire neuron.

    dv/dt = -(v - v_rest) / tau_mem + R * I + noise

    Parameters use normalised units (voltage [0,1], time in ms).
    Defaults from Gerstner & Kistler, *Spiking Neuron Models*, 2002.

    Example
    -------
    >>> neuron = StochasticLIFNeuron(v_threshold=1.0, tau_mem=20.0, noise_std=0.0)
    >>> spikes = [neuron.step(1.5) for _ in range(50)]
    >>> sum(spikes) > 0
    True
    >>> neuron.get_state()  # membrane voltage + refractory counter
    {'v': ..., 'refractory': 0}

    Process a bitstream as input current:

    >>> import numpy as np
    >>> bits = np.array([1, 0, 1, 1, 0, 1, 0, 0], dtype=np.uint8)
    >>> neuron.reset_state()
    >>> out = neuron.process_bitstream(bits, input_scale=2.0)
    >>> out.shape
    (8,)
    """

    v_rest: float = LIF_V_REST
    v_reset: float = LIF_V_RESET
    v_threshold: float = LIF_V_THRESHOLD
    tau_mem: float = LIF_TAU_MEM
    dt: float = LIF_DT
    noise_std: float = LIF_NOISE_STD
    resistance: float = LIF_RESISTANCE
    refractory_period: int = LIF_REFRACTORY_PERIOD
    seed: int | None = None
    entropy_source: Any | None = None  # Optional external entropy (e.g. Quantum)

    def __post_init__(self) -> None:
        if self.tau_mem <= 0:
            raise ValueError(f"tau_mem must be > 0, got {self.tau_mem}")
        self._rng = RNG(self.seed)
        self.v = self.v_rest
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
