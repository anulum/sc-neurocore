# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Exascale Demo

import numpy as np
import sys
import os
from typing import Any

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from sc_neurocore.accel.mpi_driver import MPIDriver
from sc_neurocore.learning.neuroevolution import SNNGeneticEvolver
from sc_neurocore.drivers.physical_twin import PhysicalTwinBridge
from sc_neurocore.analysis.explainability import SpikeToConceptMapper
from sc_neurocore.layers.vectorized_layer import VectorizedSCLayer


def run_exascale_demo() -> None:
    print("--- EXASCALE SYSTEMS DEMO ---")

    # 1. MPI
    print("\n[1] Testing MPI Driver...")
    mpi = MPIDriver()
    data = np.arange(10)
    chunk = mpi.scatter_workload(data)
    print(f"    Rank {mpi.rank} received chunk size: {len(chunk)}")

    # 2. Neuroevolution
    print("\n[2] Testing Genetic Neuroevolution...")

    def factory() -> Any:
        return VectorizedSCLayer(n_inputs=5, n_neurons=1)

    def fitness(layer: Any) -> float:
        # Target: Output 1.0 for input [1,1,1,1,1]
        out = layer.forward(np.ones(5))
        return float(out[0])  # Higher is better

    evolver = SNNGeneticEvolver(factory, fitness)
    best = evolver.evolve(generations=3)
    print(f"    Best Fitness Evolved: {fitness(best):.4f}")

    # 3. Twin
    print("\n[3] Testing Digital Twin...")
    twin = PhysicalTwinBridge()
    hw_val = twin.sync_step(0.5, 1)
    print(f"    Hardware V_mem: {hw_val:.4f}")

    # 4. Explainability
    print("\n[4] Testing Semantic Explainability...")
    mapper = SpikeToConceptMapper({0: "Apple", 1: "Banana", 2: "Cherry"})
    spikes = np.array([1, 0, 1])
    explanation = mapper.explain(spikes)
    print(f"    XAI: {explanation}")


if __name__ == "__main__":
    run_exascale_demo()
