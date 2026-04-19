# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Neuron-to-Core Mapping

"""Map network neurons to physical cores on neuromorphic hardware.

Implements greedy, balanced, and locality-aware mapping strategies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .device import DeviceSpec


@dataclass
class NeuronPlacement:
    """Placement of a single neuron on hardware."""

    neuron_id: int
    core_id: int
    local_id: int  # position within the core


class Mapper:
    """Map neurons to cores using different strategies."""

    def map_greedy(
        self,
        adjacency: np.ndarray[Any, Any],
        device: DeviceSpec,
    ) -> list[NeuronPlacement]:
        """Greedy sequential mapping: fill cores one by one.

        Simple but fast. Good baseline.
        """
        n = adjacency.shape[0]
        npc = device.neurons_per_core
        placements = []

        for i in range(n):
            core = i // npc
            local = i % npc
            placements.append(NeuronPlacement(neuron_id=i, core_id=core, local_id=local))

        return placements

    def map_balanced(
        self,
        adjacency: np.ndarray[Any, Any],
        device: DeviceSpec,
    ) -> list[NeuronPlacement]:
        """Balanced mapping: distribute neurons evenly across cores.

        Neurons are assigned round-robin to minimize load imbalance.
        """
        import math

        n = adjacency.shape[0]
        npc = device.neurons_per_core
        n_cores = math.ceil(n / npc)
        n_cores = min(n_cores, device.cores)

        placements = []
        core_counts = [0] * n_cores

        for i in range(n):
            core = i % n_cores
            placements.append(
                NeuronPlacement(
                    neuron_id=i,
                    core_id=core,
                    local_id=core_counts[core],
                )
            )
            core_counts[core] += 1

        return placements

    def map_locality(
        self,
        adjacency: np.ndarray[Any, Any],
        device: DeviceSpec,
    ) -> list[NeuronPlacement]:
        """Locality-aware mapping: cluster connected neurons on same core.

        Uses a simple greedy clustering: start from the most connected
        neuron, pack its neighbors into the same core until full.
        """
        import math

        n = adjacency.shape[0]
        npc = device.neurons_per_core
        n_cores = math.ceil(n / npc)
        n_cores = min(n_cores, device.cores)

        placed = set()
        placements_dict: dict[int, NeuronPlacement] = {}

        # Degree-ordered seed selection
        degree = np.abs(adjacency).sum(axis=1) + np.abs(adjacency).sum(axis=0)
        order = np.argsort(-degree)  # highest degree first

        current_core = 0
        current_local = 0

        for seed in order:
            if seed in placed:
                continue

            # Start new core with seed
            if current_local >= npc:
                current_core += 1
                current_local = 0
                if current_core >= n_cores:
                    break

            placements_dict[seed] = NeuronPlacement(
                neuron_id=int(seed), core_id=current_core, local_id=current_local
            )
            placed.add(int(seed))
            current_local += 1

            # Pack neighbors of seed into same core
            neighbors = np.nonzero(adjacency[seed])[0]
            neighbor_strength = np.abs(adjacency[seed, neighbors])
            sorted_neighbors = neighbors[np.argsort(-neighbor_strength)]

            for nb in sorted_neighbors:
                nb = int(nb)
                if nb in placed or current_local >= npc:
                    continue
                placements_dict[nb] = NeuronPlacement(
                    neuron_id=nb, core_id=current_core, local_id=current_local
                )
                placed.add(nb)
                current_local += 1

        # Handle any remaining unplaced neurons
        for i in range(n):
            if i not in placed:
                if current_local >= npc:
                    current_core += 1
                    current_local = 0
                placements_dict[i] = NeuronPlacement(
                    neuron_id=i, core_id=current_core, local_id=current_local
                )
                current_local += 1

        return [placements_dict[i] for i in range(n)]
