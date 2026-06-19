# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Discrete-time noisy leaky integrate-and-fire neuron

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from ..constants import (
    LIF_DT,
    LIF_NOISE_STD,
    LIF_REFRACTORY_PERIOD,
    LIF_RESISTANCE,
    LIF_TAU_MEM,
    LIF_V_RESET,
    LIF_V_REST,
    LIF_V_THRESHOLD,
)
from ..utils.rng import RNG
from .base import BaseNeuron


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
        if not np.isfinite(self.v_rest):
            raise ValueError("v_rest must be finite")
        if not np.isfinite(self.v_reset):
            raise ValueError("v_reset must be finite")
        if not np.isfinite(self.v_threshold):
            raise ValueError("v_threshold must be finite")
        if not np.isfinite(self.tau_mem) or self.tau_mem <= 0:
            raise ValueError(f"tau_mem must be > 0, got {self.tau_mem}")
        if not np.isfinite(self.dt) or self.dt <= 0:
            raise ValueError(f"dt must be > 0, got {self.dt}")
        if not np.isfinite(self.noise_std) or self.noise_std < 0:
            raise ValueError(f"noise_std must be >= 0, got {self.noise_std}")
        if not np.isfinite(self.resistance) or self.resistance < 0.0:
            raise ValueError("resistance must be finite and non-negative")
        if type(self.refractory_period) is not int or self.refractory_period < 0:
            raise ValueError("refractory_period must be a non-negative integer")
        self._rng = RNG(self.seed)
        self.v = self.v_rest
        self.refractory_counter = 0
        self.reset_state()

    def step(self, input_current: float) -> int:
        self._validate_runtime_state()
        if not np.isfinite(input_current):
            raise ValueError("input_current must be finite")
        if self.refractory_counter > 0:
            self.refractory_counter -= 1
            self.v = self.v_rest
            return 0

        # Membrane leak term
        dv_leak = -(self.v - self.v_rest) * (self.dt / self.tau_mem)

        # Input term (simple Ohm's law; you can absorb R into current)
        dv_input = self.resistance * input_current * self.dt

        # Noise term (Euler-Maruyama: sigma * sqrt(dt) * N(0,1))
        dv_noise = 0.0
        if self.noise_std > 0.0:
            sqrt_dt = self.dt**0.5
            if self.entropy_source is not None:
                dv_noise = float(self.entropy_source.sample_normal(0.0, self.noise_std * sqrt_dt))
            else:
                dv_noise = float(self._rng.normal(0.0, self.noise_std * sqrt_dt))
            if not np.isfinite(dv_noise):
                raise ValueError("noise sample must be finite")

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

    def get_state(self) -> dict[str, Any]:
        self._validate_runtime_state()
        return {"v": float(self.v), "refractory": self.refractory_counter}

    def _validate_runtime_state(self) -> None:
        if not np.isfinite(self.v):
            raise ValueError("v must be finite")
        if type(self.refractory_counter) is not int or self.refractory_counter < 0:
            raise ValueError("refractory_counter must be a non-negative integer")

    def process_bitstream(
        self, input_bits: npt.ArrayLike, input_scale: float = 1.0
    ) -> npt.NDArray[np.uint8]:
        """
        Process a bitstream (array of 0s and 1s) as input current.
        Returns an array of spikes (0s and 1s).

        input_scale: scaling factor to convert bit (0/1) to current amplitude.
        """
        bits = np.asarray(input_bits)
        if bits.ndim != 1:
            raise ValueError("input_bits must be a one-dimensional bitstream")
        if not np.all(np.isfinite(bits)):
            raise ValueError("input_bits must contain only finite values")
        if np.any((bits != 0) & (bits != 1)):
            raise ValueError("input_bits must contain only binary 0/1 values")
        if not np.isfinite(input_scale) or input_scale < 0.0:
            raise ValueError("input_scale must be finite and non-negative")

        spikes = np.zeros_like(bits, dtype=np.uint8)
        for i, bit in enumerate(bits):
            # Treat bit as current pulse of amplitude 'input_scale'
            current = bit * input_scale
            spikes[i] = self.step(current)
        return spikes
