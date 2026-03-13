# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations
from typing import Any, Optional
from dataclasses import dataclass
from typing import List, Dict
from collections.abc import Sequence
import numpy as np

from ..sources.bitstream_current_source import BitstreamCurrentSource
from ..neurons.stochastic_lif import StochasticLIFNeuron
from ..recorders.spike_recorder import BitstreamSpikeRecorder
from ..constants import (
    NEURON_SEED_OFFSET,
    DENSE_LAYER_LENGTH,
    DENSE_Y_MIN,
    DENSE_Y_MAX,
    LIF_DT,
    LIF_V_REST,
    LIF_V_RESET,
    LIF_V_THRESHOLD,
    LIF_TAU_MEM,
    LIF_LAYER_NOISE_STD,
    LIF_RESISTANCE,
)


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
    length: int = DENSE_LAYER_LENGTH
    y_min: float = DENSE_Y_MIN
    y_max: float = DENSE_Y_MAX
    dt_ms: float = LIF_DT
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
                seed = self.base_seed + NEURON_SEED_OFFSET + i

            neuron = StochasticLIFNeuron(
                v_rest=self.neuron_params.get("v_rest", LIF_V_REST),
                v_reset=self.neuron_params.get("v_reset", LIF_V_RESET),
                v_threshold=self.neuron_params.get("v_threshold", LIF_V_THRESHOLD),
                tau_mem=self.neuron_params.get("tau_mem", LIF_TAU_MEM),
                dt=self.dt_ms,
                noise_std=self.neuron_params.get("noise_std", LIF_LAYER_NOISE_STD),
                resistance=self.neuron_params.get("resistance", LIF_RESISTANCE),
                seed=seed,
            )
            self.neurons.append(neuron)
            self.recorders.append(BitstreamSpikeRecorder(dt_ms=self.dt_ms))

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
        for _ in range(T):
            I_t = self.source.step()
            for neuron, rec in zip(self.neurons, self.recorders):
                spike = neuron.step(I_t)
                rec.record(spike)

    def get_spike_trains(self) -> np.ndarray[Any, Any]:
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
