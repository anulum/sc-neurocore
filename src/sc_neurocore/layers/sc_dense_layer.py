# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Stochastic-computing dense layer of LIF neurons

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
    Stochastic-computing dense layer of LIF neurons.

    Each neuron receives SC dot-product input current and produces independent
    spike trains. ``weight_values`` may be either a single shared vector of
    length ``n_inputs`` or a dense matrix shaped ``(n_neurons, n_inputs)``.
    Software-only but fully SC-driven at the input/synapse level.

    Example
    -------
    >>> layer = SCDenseLayer(
    ...     n_neurons=4, x_inputs=[0.5, 0.3], weight_values=[0.8, 0.6],
    ...     x_min=0.0, x_max=1.0, w_min=0.0, w_max=1.0, length=256,
    ... )
    >>> layer.run(T=100)
    >>> trains = layer.get_spike_trains()
    >>> trains.shape
    (4, 100)
    """

    n_neurons: int
    x_inputs: Sequence[float]
    weight_values: Sequence[float] | Sequence[Sequence[float]]
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
    sc_mode: str = "unipolar"

    def __post_init__(self) -> None:
        if self.n_neurons < 0:
            raise ValueError("n_neurons must be non-negative.")
        self._weight_matrix, self._shared_source = self._normalise_weight_values()

        self.source = BitstreamCurrentSource(
            x_inputs=self.x_inputs,
            x_min=self.x_min,
            x_max=self.x_max,
            weight_values=self._weight_matrix[0],
            w_min=self.w_min,
            w_max=self.w_max,
            length=self.length,
            y_min=self.y_min,
            y_max=self.y_max,
            seed=self.base_seed,
            sc_mode=self.sc_mode,
        )
        self.sources: List[BitstreamCurrentSource] = [self.source]
        if not self._shared_source:
            self.sources.extend(
                BitstreamCurrentSource(
                    x_inputs=self.x_inputs,
                    x_min=self.x_min,
                    x_max=self.x_max,
                    weight_values=self._weight_matrix[i],
                    w_min=self.w_min,
                    w_max=self.w_max,
                    length=self.length,
                    y_min=self.y_min,
                    y_max=self.y_max,
                    seed=None if self.base_seed is None else self.base_seed + (10_000 * i),
                    sc_mode=self.sc_mode,
                )
                for i in range(1, self.n_neurons)
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

    def _normalise_weight_values(self) -> tuple[np.ndarray[Any, Any], bool]:
        weights = np.asarray(self.weight_values, dtype=np.float64)
        n_inputs = len(self.x_inputs)
        if weights.ndim == 1:
            if weights.shape != (n_inputs,):
                raise ValueError(
                    "1-D weight_values must have shape (n_inputs,), "
                    f"got {weights.shape} for n_inputs={n_inputs}"
                )
            if not np.all(np.isfinite(weights)):
                raise ValueError("weight_values must contain only finite values.")
            return weights.reshape(1, n_inputs), True

        if weights.ndim == 2:
            if self.n_neurons == 0:
                raise ValueError("2-D weight_values requires at least one output neuron.")
            expected = (self.n_neurons, n_inputs)
            if weights.shape != expected:
                raise ValueError(
                    f"2-D weight_values must have shape {expected}, got {weights.shape}"
                )
            if not np.all(np.isfinite(weights)):
                raise ValueError("weight_values must contain only finite values.")
            return weights, False

        raise ValueError(
            "weight_values must be either a 1-D shared vector or a 2-D "
            f"(n_neurons, n_inputs) matrix, got {weights.ndim}-D"
        )

    def reset(self) -> None:
        for source in self.sources:
            source.reset()
        for neuron, rec in zip(self.neurons, self.recorders):
            neuron.reset_state()
            rec.reset()

    def run(self, T: int) -> None:
        """
        Run the layer for T time steps, updating all neurons.

        A 1-D weight vector produces one shared SC current source for all
        neurons. A 2-D weight matrix produces one SC current source per neuron.
        """
        if self._shared_source:
            for _ in range(T):
                I_t = self.source.step()
                for neuron, rec in zip(self.neurons, self.recorders):
                    spike = neuron.step(I_t)
                    rec.record(spike)
            return

        for _ in range(T):
            currents = [source.step() for source in self.sources]
            for neuron, rec, I_t in zip(self.neurons, self.recorders, currents):
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
