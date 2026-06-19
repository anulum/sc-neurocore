# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Deep Research Demo

import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from sc_neurocore.transformers.block import StochasticTransformerBlock
from sc_neurocore.chaos.rng import ChaoticRNG
from sc_neurocore.neurons.dendritic import StochasticDendriticNeuron
from sc_neurocore.physics.heat import StochasticHeatSolver
from sc_neurocore.layers.memristive import MemristiveDenseLayer
from sc_neurocore.graphs.gnn import StochasticGraphLayer


def run_deep_research_demo() -> None:
    print("--- DEEP RESEARCH FRONTIERS DEMO ---")

    # 1. Spiking Transformer
    print("\n[1] Testing S-Former Block...")
    transformer = StochasticTransformerBlock(d_model=8, n_heads=1)
    x = np.random.random((8,))
    out = transformer.forward(x)
    print(f"    Input: {x[:4]}...")
    print(f"    Output: {out[:4]}...")

    # 2. Chaotic RNG
    print("\n[2] Testing Chaotic RNG...")
    chaos = ChaoticRNG(r=4.0)
    bits = chaos.generate_bitstream(0.5, 20)
    print(f"    Chaotic Bits: {bits}")

    # 3. Dendritic Neuron (XOR)
    print("\n[3] Testing Dendritic Neuron (XOR Logic)...")
    dend = StochasticDendriticNeuron()
    # 0,0 -> 0
    print(f"    0,0 -> {dend.step(0, 0)}")
    # 1,0 -> 1
    print(f"    1,0 -> {dend.step(1, 0)}")
    # 0,1 -> 1
    print(f"    0,1 -> {dend.step(0, 1)}")
    # 1,1 -> 0
    print(f"    1,1 -> {dend.step(1, 1)}")

    # 4. Stochastic Heat Solver
    print("\n[4] Testing Heat Equation Solver...")
    heat = StochasticHeatSolver(length=20.0, num_walkers=1000, diffusivity=0.1)
    # Start in middle
    heat.walkers[:] = 10
    heat.step()
    temp = heat.get_density(n_bins=20)
    print(f"    Temp Profile Center: {temp[8:13]}")

    # 5. Memristive Layer
    print("\n[5] Testing Memristive Layer...")
    mem = MemristiveDenseLayer(n_inputs=10, n_neurons=5, stuck_rate=0.1)
    print("    Weights corrupted with noise/stuck faults.")

    # 6. E-GNN
    print("\n[6] Testing Event-Based GNN...")
    adj = np.array([[0, 1], [1, 0]])  # 2 nodes connected
    gnn = StochasticGraphLayer(adj, n_features=4)
    feats = np.random.random((2, 4))
    g_out = gnn.forward(feats)
    print(f"    GNN Output Shape: {g_out.shape}")


if __name__ == "__main__":
    run_deep_research_demo()
