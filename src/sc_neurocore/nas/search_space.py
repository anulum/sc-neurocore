# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — NAS search space definition

"""Define the architecture search space for hardware-aware SNN NAS.

Search dimensions:
  - n_layers: number of hidden layers
  - widths: neurons per layer
  - neuron_type: per-layer neuron model
  - bitstream_length: per-layer SC precision (L)
  - delay_range: maximum synaptic delay per layer

Each architecture encodes one point in this joint space.
FPGA constraints (LUT, BRAM budgets) prune infeasible points.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Architecture:
    """One point in the NAS search space."""

    n_inputs: int
    layer_widths: list[int]
    neuron_types: list[str]
    bitstream_lengths: list[int]
    delay_ranges: list[int]
    fitness_accuracy: float = 0.0
    fitness_luts: int = 0
    fitness_energy_nj: float = 0.0
    dominates_count: int = 0

    @property
    def n_layers(self) -> int:
        """Return the number of layers encoded by this architecture."""
        return len(self.layer_widths)

    @property
    def layer_sizes(self) -> list[tuple[int, int]]:
        """Return adjacent layer dimensions for hardware cost estimation."""
        sizes: list[tuple[int, int]] = []
        prev = self.n_inputs
        for w in self.layer_widths:
            sizes.append((prev, w))
            prev = w
        return sizes

    @property
    def total_params(self) -> int:
        """Return the dense connection count across all encoded layers."""
        return sum(n_in * n_out for n_in, n_out in self.layer_sizes)


NEURON_CHOICES = [
    "StochasticLIFNeuron",
    "SCIzhikevichNeuron",
    "HomeostaticLIFNeuron",
    "FixedPointLIFNeuron",
]

WIDTH_CHOICES = [8, 16, 32, 64, 128, 256]
L_CHOICES = [32, 64, 128, 256, 512]
DELAY_CHOICES = [0, 1, 2, 4, 8]


@dataclass
class SearchSpace:
    """Configurable NAS search space.

    Parameters
    ----------
    n_inputs : int
        Input dimension.
    n_outputs : int
        Output dimension (width of final layer).
    min_layers, max_layers : int
        Range of hidden layer count.
    width_choices : list of int
        Candidate widths per layer.
    neuron_choices : list of str
        Candidate neuron models.
    L_choices : list of int
        Candidate bitstream lengths.
    delay_choices : list of int
        Candidate max-delay values.
    """

    n_inputs: int
    n_outputs: int
    min_layers: int = 1
    max_layers: int = 4
    width_choices: list[int] = field(default_factory=lambda: list(WIDTH_CHOICES))
    neuron_choices: list[str] = field(default_factory=lambda: list(NEURON_CHOICES))
    L_choices: list[int] = field(default_factory=lambda: list(L_CHOICES))
    delay_choices: list[int] = field(default_factory=lambda: list(DELAY_CHOICES))

    def random_architecture(self, rng: np.random.RandomState) -> Architecture:
        """Sample a random architecture from the space."""
        n_layers = rng.randint(self.min_layers, self.max_layers + 1)
        widths = [int(rng.choice(self.width_choices)) for _ in range(n_layers - 1)]
        widths.append(self.n_outputs)
        neurons = [str(rng.choice(self.neuron_choices)) for _ in range(n_layers)]
        lengths = [int(rng.choice(self.L_choices)) for _ in range(n_layers)]
        delays = [int(rng.choice(self.delay_choices)) for _ in range(n_layers)]
        return Architecture(
            n_inputs=self.n_inputs,
            layer_widths=widths,
            neuron_types=neurons,
            bitstream_lengths=lengths,
            delay_ranges=delays,
        )

    def mutate(self, arch: Architecture, rng: np.random.RandomState) -> Architecture:
        """Mutate one random gene in the architecture."""
        widths = list(arch.layer_widths)
        neurons = list(arch.neuron_types)
        lengths = list(arch.bitstream_lengths)
        delays = list(arch.delay_ranges)

        gene = rng.randint(0, 4)
        layer_idx = rng.randint(0, arch.n_layers)

        if gene == 0 and layer_idx < arch.n_layers - 1:
            widths[layer_idx] = int(rng.choice(self.width_choices))
        elif gene == 1:
            neurons[layer_idx] = str(rng.choice(self.neuron_choices))
        elif gene == 2:
            lengths[layer_idx] = int(rng.choice(self.L_choices))
        else:
            delays[layer_idx] = int(rng.choice(self.delay_choices))

        return Architecture(
            n_inputs=arch.n_inputs,
            layer_widths=widths,
            neuron_types=neurons,
            bitstream_lengths=lengths,
            delay_ranges=delays,
        )

    def crossover(
        self, a: Architecture, b: Architecture, rng: np.random.RandomState
    ) -> Architecture:
        """Uniform crossover between two architectures of equal layer count."""
        n = min(a.n_layers, b.n_layers)
        widths, neurons, lengths, delays = [], [], [], []
        for i in range(n):
            src = a if rng.random() < 0.5 else b
            widths.append(src.layer_widths[i])
            neurons.append(src.neuron_types[i])
            lengths.append(src.bitstream_lengths[i])
            delays.append(src.delay_ranges[i])
        return Architecture(
            n_inputs=a.n_inputs,
            layer_widths=widths,
            neuron_types=neurons,
            bitstream_lengths=lengths,
            delay_ranges=delays,
        )

    @property
    def space_size(self) -> int:
        """Approximate total architectures in the search space."""
        per_layer = (
            len(self.width_choices)
            * len(self.neuron_choices)
            * len(self.L_choices)
            * len(self.delay_choices)
        )
        total = 0
        for n in range(self.min_layers, self.max_layers + 1):
            total += per_layer**n
        return total
