#!/usr/bin/env python3
"""Example 06: Ensemble Voting and Consensus.

Demonstrates multi-agent ensemble orchestration where multiple
CognitiveOrchestrator agents vote on outputs via averaging.
"""

import numpy as np
from sc_neurocore.core import CognitiveOrchestrator
from sc_neurocore.ensembles import EnsembleOrchestrator
from sc_neurocore import VectorizedSCLayer

def main():
    print("=== SC-NeuroCore: Ensemble Consensus ===\n")

    # Create 3 agents with different random initialisations
    ensemble = EnsembleOrchestrator()

    for i in range(3):
        agent = CognitiveOrchestrator()
        layer = VectorizedSCLayer(n_inputs=4, n_neurons=2, length=512)
        agent.register_module(f"layer_{i}", layer)
        ensemble.add_agent(f"agent_{i}", agent)

    print(f"Ensemble has {len(ensemble.agents)} agents")

    # Demonstrate coordinated mission (logging output)
    ensemble.coordinated_mission("classify_pattern")

    print("\nDone.")


if __name__ == "__main__":
    main()
