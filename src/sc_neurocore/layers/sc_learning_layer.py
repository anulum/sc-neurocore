from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence, List, Optional
import numpy as np

from ..synapses.stochastic_stdp import StochasticSTDPSynapse
from ..neurons.stochastic_lif import StochasticLIFNeuron
from ..recorders.spike_recorder import BitstreamSpikeRecorder
from ..utils.bitstreams import BitstreamEncoder


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

    def run_epoch(self, input_values: Sequence[float]) -> np.ndarray:
        """
        Run one bitstream epoch (length 'length').
        """
        # 1. Encode inputs
        input_bitstreams = [
            self.input_encoders[i].encode(input_values[i]) for i in range(self.n_inputs)
        ]

        # 2. Process time steps
        epoch_spikes = np.zeros((self.n_neurons, self.length), dtype=np.uint8)

        for t in range(self.length):
            for i in range(self.n_neurons):
                neuron = self.neurons[i]
                neuron_syns = self.synapses[i]

                # Compute total input current for this neuron at time t
                current_sum = 0.0
                weight_bits = []

                for j in range(self.n_inputs):
                    pre_bit = input_bitstreams[j][t]
                    # We need a bit from the synapse.
                    # We'll use the probability to get a bit.
                    w_prob = neuron_syns[j].effective_weight_probability()
                    w_bit = 1 if np.random.random() < w_prob else 0

                    current_sum += pre_bit & w_bit
                    weight_bits.append(w_bit)

                # Step neuron
                post_spike = neuron.step(current_sum)
                epoch_spikes[i, t] = post_spike
                self.recorders[i].record(post_spike)

                # 3. Update STDP for all synapses of this neuron
                for j in range(self.n_inputs):
                    pre_bit = input_bitstreams[j][t]
                    # Use the synapse's internal logic for update (if we had it step-wise)
                    # We'll manually call potentiate/depress here to be explicit
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
