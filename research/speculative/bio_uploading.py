# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header


import numpy as np
from dataclasses import dataclass
from sc_neurocore.neurons.stochastic_lif import StochasticLIFNeuron


@dataclass
class ConnectomeEmulator:
    """
    Framework for Whole Brain Emulation (Consciousness Uploading).
    Simulates massive sparse connectomes.
    """

    n_neurons: int
    sparsity: float = 0.01

    def __post_init__(self):
        # Sparse Adjacency Matrix (Weights)
        # Using a dense mock for small scale, real use would use scipy.sparse
        self.adj = np.random.random((self.n_neurons, self.n_neurons))
        self.adj[self.adj > self.sparsity] = 0

        self.neurons = [StochasticLIFNeuron() for _ in range(self.n_neurons)]
        self.spikes = np.zeros(self.n_neurons, dtype=np.uint8)

    def step(self) -> np.ndarray:
        """
        Executes one clock cycle of the entire brain slice.
        """
        # 1. Compute incoming currents
        # I = Adj * Spikes_prev
        currents = np.dot(self.adj, self.spikes)

        # 2. Update all neurons
        new_spikes = np.zeros(self.n_neurons, dtype=np.uint8)
        for i in range(self.n_neurons):
            new_spikes[i] = self.neurons[i].step(currents[i])

        self.spikes = new_spikes
        return self.spikes
