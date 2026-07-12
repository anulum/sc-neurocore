# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Evolutionary mutation and crossover operators

"""Mutate and recombine evolutionary genomes with seeded randomness."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import numpy as np

from sc_neurocore.evo_substrate.genome import Genome


class MutationType(Enum):
    """Identify the mutation operator applied to a child genome."""

    POINT = "point"  # Gaussian perturbation
    STRUCTURAL = "structural"  # Add/remove neurons/connections
    DUPLICATION = "duplication"  # Gene duplication
    SWAP = "swap"  # Swap sub-gene blocks
    IDENTITY = "identity"  # No mutation (clone)


@dataclass
class MutationConfig:
    """Controls mutation rates and magnitudes."""

    point_rate: float = 0.2
    point_sigma: float = 0.05
    structural_rate: float = 0.05
    duplication_rate: float = 0.01
    swap_rate: float = 0.02
    max_neurons: int = 1024
    min_neurons: int = 4


class MutationEngine:
    """Applies mutations to genomes."""

    def __init__(self, config: Optional[MutationConfig] = None, rng_seed: int = 42) -> None:
        self.config = config or MutationConfig()
        self.rng = np.random.default_rng(rng_seed)

    def mutate(self, genome: Genome) -> Tuple[Genome, MutationType]:
        """Apply a random mutation and return the mutated child."""
        child = copy.deepcopy(genome)
        child.parent_id = genome.genome_id
        child.generation = genome.generation + 1
        child.identity_deep = 0.0  # New organism starts fresh

        roll = self.rng.random()
        cumulative = 0.0

        cumulative += self.config.structural_rate
        if roll < cumulative:
            self._structural_mutation(child)
            child.compute_id()
            return child, MutationType.STRUCTURAL

        cumulative += self.config.duplication_rate
        if roll < cumulative:
            self._duplication_mutation(child)
            child.compute_id()
            return child, MutationType.DUPLICATION

        cumulative += self.config.swap_rate
        if roll < cumulative:
            self._swap_mutation(child)
            child.compute_id()
            return child, MutationType.SWAP

        # Default: point mutation
        self._point_mutation(child)
        child.compute_id()
        return child, MutationType.POINT

    def _point_mutation(self, genome: Genome) -> None:
        v = genome.to_vector()
        mask = self.rng.random(len(v)) < self.config.point_rate
        noise = self.rng.normal(0, self.config.point_sigma, size=len(v))
        v[mask] += noise[mask] * (np.abs(v[mask]) + 1e-8)
        rebuilt = Genome.from_vector(v, genome.generation)
        genome.topology = rebuilt.topology
        genome.neuron = rebuilt.neuron
        genome.plasticity = rebuilt.plasticity

    def _structural_mutation(self, genome: Genome) -> None:
        delta = self.rng.choice([-2, -1, 1, 2])
        genome.topology.num_neurons = int(
            np.clip(
                genome.topology.num_neurons + delta,
                self.config.min_neurons,
                self.config.max_neurons,
            )
        )
        genome.topology.connectivity += self.rng.normal(0, 0.05)
        genome.topology.connectivity = float(np.clip(genome.topology.connectivity, 0.01, 1.0))

    def _duplication_mutation(self, genome: Genome) -> None:
        genome.topology.num_layers = min(10, genome.topology.num_layers + 1)
        genome.topology.num_neurons = min(
            self.config.max_neurons,
            int(genome.topology.num_neurons * 1.5),
        )

    def _swap_mutation(self, genome: Genome) -> None:
        genome.neuron.tau_fast, genome.neuron.tau_work = (
            genome.neuron.tau_work,
            genome.neuron.tau_fast,
        )


# ── Crossover ────────────────────────────────────────────────────────


class CrossoverEngine:
    """Uniform crossover between two parent genomes."""

    def __init__(self, rng_seed: int = 42) -> None:
        self.rng = np.random.default_rng(rng_seed)

    def crossover(self, parent_a: Genome, parent_b: Genome) -> Genome:
        """Uniform crossover: each gene drawn from either parent."""
        va = parent_a.to_vector()
        vb = parent_b.to_vector()
        mask = self.rng.random(len(va)) < 0.5
        child_v = np.where(mask, va, vb)
        child = Genome.from_vector(child_v, max(parent_a.generation, parent_b.generation) + 1)
        child.parent_id = f"{parent_a.genome_id}x{parent_b.genome_id}"
        child.compute_id()
        return child


__all__ = ["CrossoverEngine", "MutationConfig", "MutationEngine", "MutationType"]
