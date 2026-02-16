from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence, List, Optional
import numpy as np

from ..synapses.stochastic_stdp import StochasticSTDPSynapse
from ..neurons.stochastic_lif import StochasticLIFNeuron
from ..recorders.spike_recorder import BitstreamSpikeRecorder
from ..utils.bitstreams import BitstreamEncoder
from ..accel._dispatch import njit_or_python


@njit_or_python(cache=True)
def _learning_epoch_kernel(  # pragma: no cover — Numba JIT compiled
    input_bitstreams: np.ndarray,
    w_probs: np.ndarray,
    n_neurons: int,
    n_inputs: int,
    length: int,
    v_rest: np.ndarray,
    v_reset: np.ndarray,
    v_threshold: np.ndarray,
    dt_over_tau: np.ndarray,
    resistance_dt: np.ndarray,
    rand_weight: np.ndarray,
    rand_ltp: np.ndarray,
    rand_ltd: np.ndarray,
    learning_rate: float,
) -> tuple:
    """JIT inner loop for run_epoch."""
    epoch_spikes = np.zeros((n_neurons, length), dtype=np.uint8)
    v = v_rest.copy()
    w_out = w_probs.copy()

    for t in range(length):
        for i in range(n_neurons):
            current_sum = 0.0
            for j in range(n_inputs):
                pre_bit = input_bitstreams[j, t]
                w_bit = 1 if rand_weight[t, i, j] < w_out[i, j] else 0
                current_sum += pre_bit & w_bit

            # LIF step
            dv_leak = -(v[i] - v_rest[i]) * dt_over_tau[i]
            dv_input = resistance_dt[i] * current_sum
            v[i] += dv_leak + dv_input
            post_spike = 0
            if v[i] >= v_threshold[i]:
                post_spike = 1
                v[i] = v_reset[i]
            epoch_spikes[i, t] = post_spike

            # STDP update
            for j in range(n_inputs):
                pre_bit = input_bitstreams[j, t]
                if pre_bit == 1 and post_spike == 1:
                    if rand_ltp[t, i, j] < learning_rate:
                        w_out[i, j] = min(w_out[i, j] + 0.01, 1.0)
                elif pre_bit == 1 and post_spike == 0:
                    if rand_ltd[t, i, j] < learning_rate * 0.5:
                        w_out[i, j] = max(w_out[i, j] - 0.01, 0.0)

    return epoch_spikes, w_out


@dataclass
class SCLearningLayer:
    """
    An SC Dense Layer with integrated STDP learning.
    Each neuron has its own unique weights for the input vector.
    """

    n_inputs: int
    n_neurons: int
    w_min: float = 0.0
    w_max: float = 1.0
    learning_rate: float = 0.01
    length: int = 1024
    base_seed: Optional[int] = None

    def __post_init__(self) -> None:
        self.neurons: List[StochasticLIFNeuron] = []
        # synapses[neuron_idx][input_idx]
        self.synapses: List[List[StochasticSTDPSynapse]] = []
        self.recorders: List[BitstreamSpikeRecorder] = []

        self.input_encoders = [
            BitstreamEncoder(
                x_min=0,
                x_max=1,
                length=self.length,
                seed=self.base_seed + i if self.base_seed else None,
            )
            for i in range(self.n_inputs)
        ]

        for i in range(self.n_neurons):
            neuron_seed = self.base_seed + 1000 + i if self.base_seed else None
            self.neurons.append(StochasticLIFNeuron(seed=neuron_seed))
            self.recorders.append(BitstreamSpikeRecorder())

            neuron_syns = []
            for j in range(self.n_inputs):
                syn_seed = self.base_seed + 2000 + i * self.n_inputs + j if self.base_seed else None
                initial_w = np.random.uniform(self.w_min, self.w_max)
                neuron_syns.append(
                    StochasticSTDPSynapse(
                        w_min=self.w_min,
                        w_max=self.w_max,
                        w=initial_w,
                        learning_rate=self.learning_rate,
                        length=self.length,
                        seed=syn_seed,
                    )
                )
            self.synapses.append(neuron_syns)

    def _can_use_fast_path(self) -> bool:
        """Check if all neurons support the JIT path."""
        return all(n._can_use_fast_path() for n in self.neurons)

    def run_epoch(self, input_values: Sequence[float]) -> np.ndarray:
        """Run one bitstream epoch (length 'length')."""
        if self.n_inputs == 0:
            return np.zeros((self.n_neurons, self.length), dtype=np.uint8)

        # 1. Encode inputs
        input_bitstreams = np.stack(
            [self.input_encoders[i].encode(input_values[i]) for i in range(self.n_inputs)]
        )  # (n_inputs, length)

        if self._can_use_fast_path() and self.n_neurons > 0:
            w_probs = np.array([
                [self.synapses[i][j].effective_weight_probability()
                 for j in range(self.n_inputs)]
                for i in range(self.n_neurons)
            ])

            N = self.n_neurons
            v_rest = np.array([n.v_rest for n in self.neurons])
            v_reset = np.array([n.v_reset for n in self.neurons])
            v_threshold = np.array([n.v_threshold for n in self.neurons])
            dt_over_tau = np.array([n.dt / n.tau_mem for n in self.neurons])
            resistance_dt = np.array([n.resistance * n.dt for n in self.neurons])

            rand_weight = np.random.random((self.length, N, self.n_inputs))
            rand_ltp = np.random.random((self.length, N, self.n_inputs))
            rand_ltd = np.random.random((self.length, N, self.n_inputs))

            epoch_spikes, w_out = _learning_epoch_kernel(
                input_bitstreams, w_probs, N, self.n_inputs, self.length,
                v_rest, v_reset, v_threshold, dt_over_tau, resistance_dt,
                rand_weight, rand_ltp, rand_ltd, self.learning_rate,
            )

            # Apply learning delta back to synapse .w (preserves original .w when lr=0)
            w_delta = w_out - w_probs
            for i in range(N):
                for j in range(self.n_inputs):
                    if w_delta[i, j] != 0.0:
                        new_w = self.synapses[i][j].w + w_delta[i, j]
                        self.synapses[i][j].w = float(np.clip(new_w, self.w_min, self.w_max))

            for i in range(N):
                for t in range(self.length):
                    self.recorders[i].record(int(epoch_spikes[i, t]))

            return epoch_spikes

        # Fallback: original Python loop
        epoch_spikes = np.zeros((self.n_neurons, self.length), dtype=np.uint8)

        for t in range(self.length):
            for i in range(self.n_neurons):
                neuron = self.neurons[i]
                neuron_syns = self.synapses[i]

                current_sum = 0.0
                weight_bits = []

                for j in range(self.n_inputs):
                    pre_bit = input_bitstreams[j][t]
                    w_prob = neuron_syns[j].effective_weight_probability()
                    w_bit = 1 if np.random.random() < w_prob else 0

                    current_sum += pre_bit & w_bit
                    weight_bits.append(w_bit)

                post_spike = neuron.step(current_sum)
                epoch_spikes[i, t] = post_spike
                self.recorders[i].record(post_spike)

                for j in range(self.n_inputs):
                    pre_bit = input_bitstreams[j][t]
                    if pre_bit == 1 and post_spike == 1:
                        if np.random.random() < self.learning_rate:
                            neuron_syns[j]._potentiate()
                    elif pre_bit == 1 and post_spike == 0:
                        if np.random.random() < self.learning_rate * 0.5:
                            neuron_syns[j]._depress()

        return epoch_spikes

    def get_weights(self) -> np.ndarray:
        weights = np.zeros((self.n_neurons, self.n_inputs))
        for i in range(self.n_neurons):
            for j in range(self.n_inputs):
                weights[i, j] = self.synapses[i][j].w
        return weights
