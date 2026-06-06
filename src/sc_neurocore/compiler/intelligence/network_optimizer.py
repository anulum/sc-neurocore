# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Network topology optimizer

"""SNN network topology optimisation utilities.

Optimises neuron partitioning across multiple chips to minimise
inter-chip spike communication bandwidth.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TopologyPlan:
    """Multi-chip network topology optimisation result.

    Attributes
    ----------
    chip_assignment : dict[int, int]
        Neuron index → chip index.
    inter_chip_spikes : int
        Estimated inter-chip spikes per timestep.
    intra_chip_spikes : int
        Estimated intra-chip spikes per timestep.
    bandwidth_reduction : float
        Reduction vs naive assignment.
    num_chips : int
        Total chips used.
    """

    chip_assignment: dict[int, int]
    inter_chip_spikes: int
    intra_chip_spikes: int
    bandwidth_reduction: float
    num_chips: int


def optimize_network_topology(
    adjacency: dict[int, list[int]],
    *,
    num_chips: int = 2,
    neurons_per_chip: int | None = None,
) -> TopologyPlan:
    """Optimize SNN partitioning across multiple chips.

    Minimises inter-chip spike communication by grouping
    heavily-connected neurons onto the same chip.

    Parameters
    ----------
    adjacency : dict[int, list[int]]
        Neuron connectivity: source → list of targets.
    num_chips : int
        Number of available chips.
    neurons_per_chip : int, optional
        Max neurons per chip. Default: ceil(N / num_chips).

    Returns
    -------
    TopologyPlan
        Optimised chip assignment.
    """
    neurons = sorted(adjacency.keys())
    n = len(neurons)

    if neurons_per_chip is None:
        neurons_per_chip = max(1, -(-n // num_chips))  # ceil div

    # Simple greedy: assign neurons in adjacency order
    assignment: dict[int, int] = {}
    chip_counts = [0] * num_chips

    for neuron in neurons:
        # Prefer chip with most existing neighbours
        chip_scores = [0] * num_chips
        for target in adjacency.get(neuron, []):
            if target in assignment:
                chip_scores[assignment[target]] += 1

        # Find best chip with capacity
        best_chip = 0
        best_score = -1
        for c in range(num_chips):
            if chip_counts[c] < neurons_per_chip and chip_scores[c] > best_score:
                best_score = chip_scores[c]
                best_chip = c

        assignment[neuron] = best_chip
        chip_counts[best_chip] += 1

    # Count inter/intra chip spikes
    inter = 0
    intra = 0
    for src, targets in adjacency.items():
        for tgt in targets:
            if tgt in assignment:
                if assignment.get(src) != assignment.get(tgt):
                    inter += 1
                else:
                    intra += 1

    # Compare against naive (round-robin)
    naive_inter = 0
    for src, targets in adjacency.items():
        for tgt in targets:
            if src % num_chips != tgt % num_chips:
                naive_inter += 1

    reduction = 1.0 - (inter / max(1, naive_inter)) if naive_inter > 0 else 0.0

    return TopologyPlan(
        chip_assignment=assignment,
        inter_chip_spikes=inter,
        intra_chip_spikes=intra,
        bandwidth_reduction=round(reduction, 4),
        num_chips=num_chips,
    )
