# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Evolutionary population ecology

"""Model island migration, novelty, extinction, and co-evolution."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List

import numpy as np

from sc_neurocore.evo_substrate.organism import Organism


@dataclass
class Island:
    """One sub-population (deme) in an island model."""

    island_id: int
    population: List[Organism] = field(default_factory=list)
    best_fitness: float = 0.0


class IslandModel:
    """Multi-deme evolution with periodic migration."""

    def __init__(self, num_islands: int = 4, migration_rate: float = 0.1) -> None:
        self.islands = {i: Island(i) for i in range(num_islands)}
        self.migration_rate = migration_rate
        self.total_migrations: int = 0

    def add_organism(self, island_id: int, organism: Organism) -> None:
        """Append an organism to the selected island population."""
        self.islands[island_id].population.append(organism)

    def migrate(self, rng: np.random.Generator) -> int:
        """Migrate best organisms between random island pairs."""
        ids = list(self.islands.keys())
        if len(ids) < 2:
            return 0
        migrations = 0
        for src_id in ids:
            if rng.random() < self.migration_rate:
                dst_id = rng.choice([i for i in ids if i != src_id])
                src = self.islands[src_id]
                if src.population:
                    migrant = copy.deepcopy(src.population[0])
                    self.islands[dst_id].population.append(migrant)
                    migrations += 1
        self.total_migrations += migrations
        return migrations

    @property
    def total_population(self) -> int:
        """Return the number of organisms across every island."""
        return sum(len(isl.population) for isl in self.islands.values())


class NoveltyArchive:
    """Behavioural novelty archive for novelty search."""

    def __init__(self, k_nearest: int = 5, threshold: float = 0.1) -> None:
        self.k_nearest = k_nearest
        self.threshold = threshold
        self.archive: List[np.ndarray[Any, Any]] = []

    def novelty_score(self, behaviour: np.ndarray[Any, Any]) -> float:
        """Return mean distance to the nearest archived behaviours."""
        if not self.archive:
            return 1.0
        dists = [float(np.linalg.norm(behaviour - a)) for a in self.archive]
        dists.sort()
        k = min(self.k_nearest, len(dists))
        return float(np.mean(dists[:k]))

    def maybe_add(self, behaviour: np.ndarray[Any, Any]) -> bool:
        """Archive a copied behaviour when its novelty exceeds the threshold."""
        score = self.novelty_score(behaviour)
        if score > self.threshold:
            self.archive.append(behaviour.copy())
            return True
        return False

    @property
    def size(self) -> int:
        """Return the number of archived behaviour vectors."""
        return len(self.archive)


class ExtinctionDetector:
    """Detects population stagnation and triggers extinction events."""

    def __init__(self, stagnation_gens: int = 10, kill_fraction: float = 0.9) -> None:
        self.stagnation_gens = stagnation_gens
        self.kill_fraction = kill_fraction
        self._best_history: List[float] = []
        self.extinction_count: int = 0

    def check(self, best_fitness: float) -> bool:
        """Record fitness and report whether recent progress stagnated."""
        self._best_history.append(best_fitness)
        if len(self._best_history) < self.stagnation_gens:
            return False
        recent = self._best_history[-self.stagnation_gens :]
        improvement = max(recent) - min(recent)
        if improvement < 1e-6:
            self.extinction_count += 1
            return True
        return False

    def apply(self, population: List[Organism], rng: np.random.Generator) -> int:
        """Kill kill_fraction of population randomly."""
        n_kill = int(len(population) * self.kill_fraction)
        indices = rng.choice(len(population), size=min(n_kill, len(population)), replace=False)
        killed = 0
        for i in sorted(indices, reverse=True):
            population[i].alive = False
            killed += 1
        return killed


# ── Co-Evolution (Predator-Prey) ────────────────────────────


@dataclass
class CoevoRole(Enum):
    """Identify an organism's role in a co-evolutionary interaction."""

    PREDATOR = "predator"
    PREY = "prey"
    SYMBIONT = "symbiont"


@dataclass
class CoevoOrganism:
    """Organism with a co-evolutionary role."""

    organism: Organism
    role: CoevoRole
    interaction_score: float = 0.0


class CoevolutionArena:
    """Runs predator-prey or symbiotic co-evolution."""

    def __init__(self) -> None:
        self.predators: List[CoevoOrganism] = []
        self.prey: List[CoevoOrganism] = []

    def add_predator(self, organism: Organism) -> None:
        """Add an organism to the predator population."""
        self.predators.append(CoevoOrganism(organism, CoevoRole.PREDATOR))

    def add_prey(self, organism: Organism) -> None:
        """Add an organism to the prey population."""
        self.prey.append(CoevoOrganism(organism, CoevoRole.PREY))

    def evaluate_interactions(self) -> Dict[str, float]:
        """Evaluate predator-prey fitness from pairwise interactions."""
        results = {}
        for pred in self.predators:
            score = sum(
                1.0
                for prey in self.prey
                if pred.organism.genome.topology.num_neurons
                > prey.organism.genome.topology.num_neurons
            )
            pred.interaction_score = score / max(1, len(self.prey))
            results[pred.organism.genome.genome_id] = pred.interaction_score
        for prey_org in self.prey:
            score = sum(
                1.0
                for pred in self.predators
                if prey_org.organism.genome.topology.connectivity
                < pred.organism.genome.topology.connectivity
            )
            prey_org.interaction_score = score / max(1, len(self.predators))
            results[prey_org.organism.genome.genome_id] = prey_org.interaction_score
        return results

    @property
    def total_organisms(self) -> int:
        """Return the combined predator and prey population."""
        return len(self.predators) + len(self.prey)


__all__ = [
    "CoevoOrganism",
    "CoevoRole",
    "CoevolutionArena",
    "ExtinctionDetector",
    "Island",
    "IslandModel",
    "NoveltyArchive",
]
