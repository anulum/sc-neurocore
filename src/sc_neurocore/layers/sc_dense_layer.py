from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence, List, Optional, Dict, Any
import numpy as np

from ..sources.bitstream_current_source import BitstreamCurrentSource
from ..neurons.stochastic_lif import StochasticLIFNeuron
from ..recorders.spike_recorder import BitstreamSpikeRecorder
from ..accel._dispatch import njit_or_python


# ---- Numba-accelerated dense-layer kernel ---------------------------------
@njit_or_python(cache=True)
def _dense_run_kernel(  # pragma: no cover — Numba JIT compiled
    currents: np.ndarray,
    n_neurons: int,
    v_rest: np.ndarray,
    v_reset: np.ndarray,
    v_threshold: np.ndarray,
    dt_over_tau: np.ndarray,
    resistance_dt: np.ndarray,
) -> np.ndarray:
    """
    Run T steps for N neurons sharing the same current sequence.

    Args:
        currents: shape (T,) — shared input current at each step.
        n_neurons: number of neurons.
        v_rest .. resistance_dt: per-neuron parameter arrays, each shape (N,).

    Returns:
        spikes: shape (N, T) uint8 array.
    """
    T = currents.shape[0]
    spikes = np.zeros((n_neurons, T), dtype=np.uint8)
    v = v_rest.copy()
    for t in range(T):
        I_t = currents[t]
        for n in range(n_neurons):
            dv_leak = -(v[n] - v_rest[n]) * dt_over_tau[n]
            dv_input = resistance_dt[n] * I_t
            v[n] += dv_leak + dv_input
            if v[n] >= v_threshold[n]:
                spikes[n, t] = 1
                v[n] = v_reset[n]
    return spikes


@dataclass
class SCDenseLayer:
    """
    Simple stochastic-computing "dense layer" of LIF neurons.

    - Each neuron shares the same multi-channel BitstreamCurrentSource
      (same inputs + weights for now, can be diversified later).
    - Each neuron has its own stochastic LIF parameters and RNG seed.
    - We simulate T time steps and collect spike trains for all neurons.

    This is software-only but fully SC-driven at the input/synapse level.
    """

    n_neurons: int
    x_inputs: Sequence[float]
    weight_values: Sequence[float]
    x_min: float
    x_max: float
    w_min: float
    w_max: float
    length: int = 2048
    y_min: float = 0.0
    y_max: float = 0.1
    dt_ms: float = 1.0
    neuron_params: Optional[Dict[str, Any]] = None
    base_seed: Optional[int] = None

    def __post_init__(self) -> None:
        if len(self.x_inputs) != len(self.weight_values):
            raise ValueError("x_inputs and weight_values must have same length.")

        # Shared SC current source for now (can be extended to per-neuron later)
        self.source = BitstreamCurrentSource(
            x_inputs=self.x_inputs,
            x_min=self.x_min,
            x_max=self.x_max,
            weight_values=self.weight_values,
            w_min=self.w_min,
            w_max=self.w_max,
            length=self.length,
            y_min=self.y_min,
            y_max=self.y_max,
            seed=self.base_seed,
        )

        # Build neurons
        if self.neuron_params is None:
            self.neuron_params = {}

        self.neurons: List[StochasticLIFNeuron] = []
        self.recorders: List[BitstreamSpikeRecorder] = []
        for i in range(self.n_neurons):
            # Give each neuron its own seed so they don't behave identically
            seed = None
            if self.base_seed is not None:
                seed = self.base_seed + 10000 + i

            neuron = StochasticLIFNeuron(
                v_rest=self.neuron_params.get("v_rest", 0.0),
                v_reset=self.neuron_params.get("v_reset", 0.0),
                v_threshold=self.neuron_params.get("v_threshold", 1.0),
                tau_mem=self.neuron_params.get("tau_mem", 20.0),
                dt=self.dt_ms,
                noise_std=self.neuron_params.get("noise_std", 0.02),
                resistance=self.neuron_params.get("resistance", 1.0),
                seed=seed,
            )
            self.neurons.append(neuron)
            self.recorders.append(BitstreamSpikeRecorder(dt_ms=self.dt_ms))

    def _can_use_fast_path(self) -> bool:
        """All neurons must have no noise and no refractory for JIT path."""
        return all(n._can_use_fast_path() for n in self.neurons)

    def reset(self) -> None:
        self.source.reset()
        for neuron, rec in zip(self.neurons, self.recorders):
            neuron.reset_state()
            rec.reset()

    def run(self, T: int) -> None:
        """
        Run the layer for T time steps, updating all neurons.

        The current I_t is shared across all neurons (common input
        processed through SC dot-product). Neurons differ by their
        internal noise and parameters.
        """
        if self._can_use_fast_path() and self.n_neurons > 0:
            # Pre-compute all T currents
            currents = np.empty(T, dtype=np.float64)
            for t in range(T):
                currents[t] = self.source.step()

            # Pack neuron parameters into arrays
            N = self.n_neurons
            v_rest = np.array([n.v_rest for n in self.neurons])
            v_reset = np.array([n.v_reset for n in self.neurons])
            v_threshold = np.array([n.v_threshold for n in self.neurons])
            dt_over_tau = np.array([n.dt / n.tau_mem for n in self.neurons])
            resistance_dt = np.array([n.resistance * n.dt for n in self.neurons])

            spikes = _dense_run_kernel(
                currents, N, v_rest, v_reset, v_threshold, dt_over_tau, resistance_dt
            )

            # Feed results into recorders
            for i in range(N):
                for t in range(T):
                    self.recorders[i].record(int(spikes[i, t]))
            return

        # Fallback: original Python loop
        for _ in range(T):
            I_t = self.source.step()
            for neuron, rec in zip(self.neurons, self.recorders):
                spike = neuron.step(I_t)
                rec.record(spike)

    def get_spike_trains(self) -> np.ndarray:
        """
        Return spike matrix of shape (n_neurons, T).
        """
        if not self.recorders:
            return np.zeros((0, 0), dtype=np.uint8)

        T = len(self.recorders[0].spikes)
        spikes = np.zeros((self.n_neurons, T), dtype=np.uint8)
        for i, rec in enumerate(self.recorders):
            spikes[i] = rec.as_array()
        return spikes

    def summary(self) -> Dict[str, Any]:
        """
        Return firing statistics for each neuron.
        """
        stats = []
        for i, rec in enumerate(self.recorders):
            stats.append(
                {
                    "neuron": i,
                    "total_spikes": rec.total_spikes(),
                    "firing_rate_hz": rec.firing_rate_hz(),
                }
            )
        return {
            "n_neurons": self.n_neurons,
            "stats": stats,
            "avg_firing_rate_hz": float(
                np.mean([s["firing_rate_hz"] for s in stats]) if stats else 0.0
            ),
        }
