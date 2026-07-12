# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Evolutionary selection and survivor regulation

"""Select, rank, preserve, and regulate evolutionary organisms."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from sc_neurocore.evo_substrate.fitness import FitnessResult
from sc_neurocore.evo_substrate.genome import Genome
from sc_neurocore.evo_substrate.organism import Organism


class HallOfFame:
    """Maintains the top-N organisms across all generations."""

    def __init__(self, max_size: int = 10) -> None:
        self.max_size = max_size
        self.entries: List[Tuple[float, Genome]] = []  # (fitness, genome)

    def update(self, organism: Organism) -> bool:
        """Insert a fitted organism and retain the highest-ranked entries."""
        if organism.fitness is None:
            return False
        fit = organism.fitness.composite
        self.entries.append((fit, copy.deepcopy(organism.genome)))
        self.entries.sort(key=lambda x: x[0], reverse=True)
        if len(self.entries) > self.max_size:
            self.entries = self.entries[: self.max_size]
        return True

    @property
    def best_fitness(self) -> float:
        """Return the highest retained composite fitness, or zero when empty."""
        return self.entries[0][0] if self.entries else 0.0

    @property
    def size(self) -> int:
        """Return the number of retained hall-of-fame entries."""
        return len(self.entries)


class TournamentSelector:
    """Tournament selection with configurable pressure."""

    def __init__(self, tournament_size: int = 3) -> None:
        self.tournament_size = tournament_size

    def select(self, population: List[Organism], rng: np.random.Generator) -> Organism:
        """Return the highest-fitness member of one random tournament.

        Raises
        ------
        ValueError
            If the population is empty.
        """
        if not population:
            raise ValueError("Tournament selection requires a non-empty population")
        candidates = rng.choice(
            len(population),
            size=min(self.tournament_size, len(population)),
            replace=False,
        )
        first_idx = int(candidates[0])
        best = population[first_idx]
        best_fit = best.fitness.composite if best.fitness else 0.0
        for idx in candidates:
            org = population[int(idx)]
            fit = org.fitness.composite if org.fitness else 0.0
            if fit > best_fit:
                best_fit = fit
                best = org
        return best

    def select_n(
        self, population: List[Organism], n: int, rng: np.random.Generator
    ) -> List[Organism]:
        """Run independent tournaments and return the requested selections."""
        return [self.select(population, rng) for _ in range(n)]


# ── Multi-Objective Pareto Front ───────────────────────────


def dominates(a: FitnessResult, b: FitnessResult) -> bool:
    """Return whether the first result Pareto-dominates the second."""
    vals_a = [a.accuracy, a.energy_score, a.latency_score]
    vals_b = [b.accuracy, b.energy_score, b.latency_score]
    at_least_one_better = False
    for va, vb in zip(vals_a, vals_b):
        if va < vb:
            return False
        if va > vb:
            at_least_one_better = True
    return at_least_one_better


class ParetoFront:
    """Maintains a non-dominated Pareto front."""

    def __init__(self) -> None:
        self.front: List[Organism] = []

    def update(self, organism: Organism) -> bool:
        """Insert an organism when no retained member dominates its fitness."""
        if organism.fitness is None:
            return False
        dominated_by = [
            o for o in self.front if o.fitness and dominates(o.fitness, organism.fitness)
        ]
        if dominated_by:
            return False
        self.front = [
            o for o in self.front if not (o.fitness and dominates(organism.fitness, o.fitness))
        ]
        self.front.append(organism)
        return True

    @property
    def size(self) -> int:
        """Return the number of non-dominated organisms."""
        return len(self.front)


# ── Age-Based Regulation ───────────────────────────────────


class AgeRegulator:
    """Culls organisms that exceed a maximum lifespan."""

    def __init__(self, max_age: int = 20) -> None:
        self.max_age = max_age

    def apply(self, population: List[Organism], current_generation: int) -> int:
        """Mark over-age organisms dead and return the number culled."""
        killed = 0
        for org in population:
            age = current_generation - org.birth_generation
            if age > self.max_age:
                org.alive = False
                killed += 1
        return killed


# ── Genome Bloat Control ───────────────────────────────────


@dataclass
class BloatMetrics:
    """Measures genome complexity for bloat detection."""

    total_params: int
    neuron_count: int
    layer_count: int
    connection_count: int
    bloat_score: float = 0.0

    @property
    def is_bloated(self) -> bool:
        """Return whether complexity exceeds the baseline bloat score."""
        return self.bloat_score > 1.0


def compute_bloat(genome: Genome, baseline_neurons: int = 16) -> BloatMetrics:
    """Compute bloat relative to a baseline."""
    n = genome.topology.num_neurons
    l = genome.topology.num_layers
    conn = int(n * n * genome.topology.connectivity)
    total = n * 8 + l + conn  # rough param count
    baseline = baseline_neurons * 8 + 2 + int(baseline_neurons**2 * 0.3)
    score = total / max(1, baseline)
    return BloatMetrics(total, n, l, conn, score)


class BloatPenalizer:
    """Penalizes fitness for bloated genomes."""

    def __init__(self, penalty_weight: float = 0.1, threshold: float = 2.0) -> None:
        self.penalty_weight = penalty_weight
        self.threshold = threshold

    def penalize(self, fitness: float, genome: Genome) -> float:
        """Apply a bounded multiplicative penalty above the bloat threshold."""
        bm = compute_bloat(genome)
        if bm.bloat_score > self.threshold:
            excess = bm.bloat_score - self.threshold
            return fitness * max(0.1, 1.0 - self.penalty_weight * excess)
        return fitness


__all__ = [
    "AgeRegulator",
    "BloatMetrics",
    "BloatPenalizer",
    "HallOfFame",
    "ParetoFront",
    "TournamentSelector",
    "compute_bloat",
    "dominates",
]
