# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Evolutionary statistics and genome comparison

"""Track generation fitness, genome differences, and complexity trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from sc_neurocore.evo_substrate.genome import Genome
from sc_neurocore.evo_substrate.organism import Organism


@dataclass
class GenerationStats:
    """Statistics for one generation."""

    generation: int
    population_size: int
    best_fitness: float
    mean_fitness: float
    diversity: float
    species_count: int = 0
    extinctions: int = 0


class EvoStatisticsTracker:
    """Records per-generation statistics for analytics."""

    def __init__(self) -> None:
        self.history: List[GenerationStats] = []

    def record(self, stats: GenerationStats) -> None:
        """Append statistics for one completed generation."""
        self.history.append(stats)

    @property
    def generations_tracked(self) -> int:
        """Return the number of recorded generations."""
        return len(self.history)

    @property
    def fitness_trajectory(self) -> List[float]:
        """Return best fitness in generation order."""
        return [s.best_fitness for s in self.history]

    @property
    def diversity_trajectory(self) -> List[float]:
        """Return population diversity in generation order."""
        return [s.diversity for s in self.history]

    def improvement_rate(self) -> float:
        """Return best-fitness change from the first to latest generation."""
        if len(self.history) < 2:
            return 0.0
        return self.history[-1].best_fitness - self.history[0].best_fitness


# ── Genome Diff / Comparison ───────────────────────────────


@dataclass
class GenomeDiff:
    """Structural diff between two genomes."""

    neuron_delta: int
    layer_delta: int
    connectivity_delta: float
    tau_fast_delta: float
    tau_deep_delta: float
    total_param_changes: int

    @property
    def is_identical(self) -> bool:
        """Return whether the canonical vectors contain no changed values."""
        return self.total_param_changes == 0


def genome_diff(a: Genome, b: Genome) -> GenomeDiff:
    """Compute structural diff between two genomes."""
    va, vb = a.to_vector(), b.to_vector()
    changes = int(np.sum(np.abs(va - vb) > 1e-8))
    return GenomeDiff(
        neuron_delta=b.topology.num_neurons - a.topology.num_neurons,
        layer_delta=b.topology.num_layers - a.topology.num_layers,
        connectivity_delta=b.topology.connectivity - a.topology.connectivity,
        tau_fast_delta=b.neuron.tau_fast - a.neuron.tau_fast,
        tau_deep_delta=b.neuron.tau_deep - a.neuron.tau_deep,
        total_param_changes=changes,
    )


# ── Open-Ended Complexity Metric ───────────────────────────


def genome_complexity(genome: Genome) -> float:
    """Measure evolved complexity (information-theoretic)."""
    v = genome.to_vector()
    v_norm = v / (np.abs(v).max() + 1e-10)
    v_pos = np.abs(v_norm) + 1e-10
    v_pos = v_pos / v_pos.sum()
    entropy = -float(np.sum(v_pos * np.log2(v_pos)))
    topology_complexity = (
        genome.topology.num_neurons * genome.topology.num_layers * genome.topology.connectivity
    )
    return float(entropy + np.log2(1 + topology_complexity))


class ComplexityTracker:
    """Tracks population complexity over generations."""

    def __init__(self) -> None:
        self.history: List[Tuple[int, float, float]] = []  # (gen, mean, max)

    def record(self, generation: int, population: List[Organism]) -> None:
        """Append mean and maximum complexity for a non-empty population."""
        if not population:
            return
        complexities = [genome_complexity(o.genome) for o in population]
        self.history.append(
            (
                generation,
                float(np.mean(complexities)),
                float(np.max(complexities)),
            )
        )

    @property
    def mean_trajectory(self) -> List[float]:
        """Return mean genome complexity in generation order."""
        return [h[1] for h in self.history]

    @property
    def is_complexifying(self) -> bool:
        """Return whether mean complexity increased over three or more samples."""
        if len(self.history) < 3:
            return False
        return self.history[-1][1] > self.history[0][1]


__all__ = [
    "ComplexityTracker",
    "EvoStatisticsTracker",
    "GenerationStats",
    "GenomeDiff",
    "genome_complexity",
    "genome_diff",
]
