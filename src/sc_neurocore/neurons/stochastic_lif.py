from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any

import numpy as np

from .base import BaseNeuron
from ..utils.rng import RNG
from ..accel._dispatch import njit_or_python


# ---- Numba-accelerated LIF kernels (no-noise, no-refractory path) ---------
@njit_or_python(cache=True)
def _lif_bitstream_kernel(  # pragma: no cover — Numba JIT compiled
    input_bits: np.ndarray,
    input_scale: float,
    v_rest: float,
    v_reset: float,
    v_threshold: float,
    dt_over_tau: float,
    resistance_dt: float,
) -> np.ndarray:
    """Pure-arithmetic LIF step over a bitstream (no noise, no refractory)."""
    n = input_bits.shape[0]
    spikes = np.zeros(n, dtype=np.uint8)
    v = v_rest
    for i in range(n):
        current = input_bits[i] * input_scale
        dv_leak = -(v - v_rest) * dt_over_tau
        dv_input = resistance_dt * current
        v += dv_leak + dv_input
        if v >= v_threshold:
            spikes[i] = 1
            v = v_reset
    return spikes


@njit_or_python(cache=True)
def _lif_step_array_kernel(  # pragma: no cover — Numba JIT compiled
    currents: np.ndarray,
    v_rest: float,
    v_reset: float,
    v_threshold: float,
    dt_over_tau: float,
    resistance_dt: float,
) -> np.ndarray:
    """Step a LIF neuron with an array of float currents (no noise)."""
    n = currents.shape[0]
    spikes = np.zeros(n, dtype=np.uint8)
    v = v_rest
    for i in range(n):
        dv_leak = -(v - v_rest) * dt_over_tau
        dv_input = resistance_dt * currents[i]
        v += dv_leak + dv_input
        if v >= v_threshold:
            spikes[i] = 1
            v = v_reset
    return spikes


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

    def _can_use_fast_path(self) -> bool:
        """Check if the neuron can use the JIT-accelerated kernel."""
        return self.noise_std == 0.0 and self.entropy_source is None and self.refractory_period == 0

    def step_array(self, currents: np.ndarray) -> np.ndarray:
        """
        Step the neuron over an array of float currents.

        Uses JIT kernel when noise/refractory are disabled, falls back to
        the per-step Python loop otherwise.
        """
        currents = np.asarray(currents, dtype=np.float64).ravel()
        if self._can_use_fast_path():
            spikes = _lif_step_array_kernel(
                currents,
                self.v_rest,
                self.v_reset,
                self.v_threshold,
                self.dt / self.tau_mem,
                self.resistance * self.dt,
            )
            # Replay final membrane state so self.v stays consistent
            v = self.v_rest
            for i in range(len(currents)):
                v += (
                    -(v - self.v_rest) * (self.dt / self.tau_mem)
                    + self.resistance * currents[i] * self.dt
                )
                if v >= self.v_threshold:
                    v = self.v_reset
            self.v = v
            return spikes

        # Fallback: per-step Python loop
        spikes = np.zeros(len(currents), dtype=np.uint8)
        for i in range(len(currents)):
            spikes[i] = self.step(currents[i])
        return spikes

    def process_bitstream(self, input_bits: np.ndarray, input_scale: float = 1.0) -> np.ndarray:
        """
        Process a bitstream (array of 0s and 1s) as input current.
        Returns an array of spikes (0s and 1s).

        input_scale: scaling factor to convert bit (0/1) to current amplitude.
        """
        if self._can_use_fast_path():
            spikes = _lif_bitstream_kernel(
                np.asarray(input_bits, dtype=np.float64),
                float(input_scale),
                self.v_rest,
                self.v_reset,
                self.v_threshold,
                self.dt / self.tau_mem,
                self.resistance * self.dt,
            )
            # Sync final membrane potential
            v = self.v_rest
            for i in range(len(input_bits)):
                current = input_bits[i] * input_scale
                v += (
                    -(v - self.v_rest) * (self.dt / self.tau_mem)
                    + self.resistance * current * self.dt
                )
                if v >= self.v_threshold:
                    v = self.v_reset
            self.v = v
            return spikes

        # Fallback: original Python loop
        spikes = np.zeros_like(input_bits, dtype=np.uint8)
        for i, bit in enumerate(input_bits):
            current = bit * input_scale
            spikes[i] = self.step(current)
        return spikes
