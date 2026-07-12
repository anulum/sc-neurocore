# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Evolutionary replication workflow

"""Compose evaluation, selection, mutation, safety, and population renewal."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np

from sc_neurocore.evo_substrate.ecology import ExtinctionDetector
from sc_neurocore.evo_substrate.fitness import FitnessEvaluator
from sc_neurocore.evo_substrate.genome import Genome
from sc_neurocore.evo_substrate.lineage import LineageTracker
from sc_neurocore.evo_substrate.organism import Organism
from sc_neurocore.evo_substrate.safety import (
    FormalSafetyGuard,
    RuntimeFaultCheck,
    RuntimeFaultConfig,
)
from sc_neurocore.evo_substrate.selection import (
    AgeRegulator,
    BloatPenalizer,
    HallOfFame,
    ParetoFront,
    TournamentSelector,
)
from sc_neurocore.evo_substrate.speciation import population_diversity, shared_fitness
from sc_neurocore.evo_substrate.statistics import EvoStatisticsTracker, GenerationStats
from sc_neurocore.evo_substrate.variation import CrossoverEngine, MutationEngine
from sc_neurocore.fault_injection import (
    DegradationAction,
    DegradationPlan,
    GracefulDegradationPolicy,
)


class ReplicationEngine:
    """Manages organism reproduction, mutation, and deployment.

    Selection → Replication → Mutation → Safety Check → Deploy
    """

    def __init__(
        self,
        mutation_engine: Optional[MutationEngine] = None,
        crossover_engine: Optional[CrossoverEngine] = None,
        fitness_evaluator: Optional[FitnessEvaluator] = None,
        max_population: int = 32,
        elitism: int = 1,
        industrial_mode: bool = True,
        runtime_fault_config: Optional[RuntimeFaultConfig] = None,
        degradation_policy: Optional[GracefulDegradationPolicy] = None,
    ) -> None:
        self.mutator = mutation_engine or MutationEngine()
        self.crossover = crossover_engine or CrossoverEngine()
        self.evaluator = fitness_evaluator or FitnessEvaluator()
        self.max_population = max_population
        self.elitism = elitism
        self.population: List[Organism] = []
        self.graveyard: List[Organism] = []
        self.generation: int = 0
        self.total_replications: int = 0
        self.lineage = LineageTracker()
        self.runtime_fault_config = runtime_fault_config
        self.degradation_policy = degradation_policy or GracefulDegradationPolicy()

        # Industrial Features
        self.industrial_mode = industrial_mode
        self.tournament = TournamentSelector()
        self.safety_guard = FormalSafetyGuard()
        self.age_regulator = AgeRegulator(max_age=20)
        self.bloat_penalizer = BloatPenalizer()
        self.extinction_detector = ExtinctionDetector(stagnation_gens=10, kill_fraction=0.9)
        self.hall_of_fame = HallOfFame()
        self.pareto_front = ParetoFront()
        self.stats_tracker = EvoStatisticsTracker()

    def seed(self, genome: Genome) -> Organism:
        """Seed the population with an initial organism."""
        genome.compute_id()
        org = Organism(genome=genome, birth_generation=0)
        self.population.append(org)
        self.lineage.record(org, "seed")
        return org

    def replicate(self, parent: Organism) -> Optional[Organism]:
        """Create a mutated child from a parent."""
        child_genome, mut_type = self.mutator.mutate(parent.genome)

        if self.industrial_mode:
            result = self.safety_guard.check(child_genome)
            if not result.passed:
                return None

        child = Organism(
            genome=child_genome,
            birth_generation=self.generation,
        )
        self.total_replications += 1
        self.lineage.record(child, mut_type.value)
        if len(self.population) < self.max_population:
            self.population.append(child)
        return child

    def replicate_crossover(self, parent_a: Organism, parent_b: Organism) -> Optional[Organism]:
        """Create a child via crossover of two parents."""
        child_genome = self.crossover.crossover(parent_a.genome, parent_b.genome)

        if self.industrial_mode:
            result = self.safety_guard.check(child_genome)
            if not result.passed:
                return None

        child = Organism(
            genome=child_genome,
            birth_generation=self.generation,
        )
        self.total_replications += 1
        self.lineage.record(child, "crossover")
        if len(self.population) < self.max_population:
            self.population.append(child)
        return child

    def evaluate_all(self, metrics_fn: Callable[[Genome], Dict[str, float]]) -> None:
        """Evaluate fitness for all living organisms."""
        for org in self.population:
            if org.alive:
                metrics = metrics_fn(org.genome)
                org.fitness = self.evaluator.evaluate(org.genome, metrics)

                if self.industrial_mode:
                    org.fitness.composite = self.bloat_penalizer.penalize(
                        org.fitness.composite, org.genome
                    )
                    org.fitness.composite = shared_fitness(org, self.population)
                    if self.runtime_fault_config is not None:
                        self.verify_runtime_faults(org, self.runtime_fault_config)
                    self.hall_of_fame.update(org)
                    self.pareto_front.update(org)

    def verify_runtime_faults(
        self,
        organism: Organism,
        config: Optional[RuntimeFaultConfig] = None,
    ) -> RuntimeFaultCheck:
        """Run seeded runtime fault diagnosis and apply bounded degradation."""
        cfg = config or self.runtime_fault_config or RuntimeFaultConfig()
        streams = self._runtime_bitstreams(organism.genome, cfg)
        replay_seed = int(organism.genome.weight_seed + cfg.seed_offset)
        plan = self.degradation_policy.evaluate(
            streams,
            layer_id=organism.genome.genome_id or "unidentified",
            fault_model=cfg.fault_model,
            ber=cfg.ber,
            seed=replay_seed,
        )
        check = RuntimeFaultCheck.from_plan(organism, plan)
        organism.runtime_fault_checks.append(check)
        self._apply_runtime_fault_plan(organism, plan, cfg)
        return check

    def _runtime_bitstreams(
        self, genome: Genome, config: RuntimeFaultConfig
    ) -> np.ndarray[Any, Any]:
        neurons = max(1, min(config.sample_neurons, genome.topology.num_neurons))
        length = max(1, genome.topology.bitstream_length)
        rng = np.random.default_rng(int(genome.weight_seed + config.seed_offset))
        return (rng.random((neurons, length)) < 0.5).astype(np.uint8)

    def _apply_runtime_fault_plan(
        self,
        organism: Organism,
        plan: DegradationPlan,
        config: RuntimeFaultConfig,
    ) -> None:
        if plan.action == DegradationAction.NOMINAL:
            return
        organism.genome.topology.bitstream_length = plan.recommended_bitstream_length
        organism.genome.compute_id()
        if organism.fitness is None:
            return
        if plan.action == DegradationAction.REPLAY_WITH_SEED:
            organism.fitness.composite *= config.fitness_penalty_on_replay
        else:
            organism.fitness.composite *= config.fitness_penalty_on_extend

    def select_and_cull(self, survival_fraction: float = 0.5) -> int:
        """Select fittest organisms, cull the rest. Elitism preserved."""
        if self.industrial_mode:
            killed = self.age_regulator.apply(self.population, self.generation)
            if self.extinction_detector.check(self.best_fitness):
                killed += self.extinction_detector.apply(self.population, self.mutator.rng)

        alive = [o for o in self.population if o.alive and o.fitness is not None]
        alive.sort(
            key=lambda o: o.fitness.composite if o.fitness is not None else float("-inf"),
            reverse=True,
        )

        cutoff = max(self.elitism + 1, int(len(alive) * survival_fraction))
        killed = 0
        for org in alive[cutoff:]:
            org.alive = False
            self.graveyard.append(org)
            killed += 1

        self.population = [o for o in self.population if o.alive]
        return killed

    def evolve_generation(self, metrics_fn: Callable[[Genome], Dict[str, float]]) -> Dict[str, Any]:
        """Run one full evolutionary generation."""
        self.generation += 1

        # 1. Evaluate
        self.evaluate_all(metrics_fn)

        # 2. Select + cull
        killed = self.select_and_cull()

        # 3. Replicate from survivors
        survivors = list(self.population)
        children_created = 0

        for i in range(len(survivors)):
            if len(self.population) >= self.max_population:
                break

            parent: Optional[Organism]
            partner: Optional[Organism]
            if self.industrial_mode:
                parent = self.tournament.select(survivors, self.mutator.rng)
                partner = self.tournament.select(survivors, self.mutator.rng)
            else:
                parent = survivors[i]
                partner = survivors[(i + 1) % len(survivors)] if len(survivors) > 1 else None

            if parent is None:
                continue

            child_added: Optional[Organism] = None
            if partner and self.mutator.rng.random() < 0.3:
                child_added = self.replicate_crossover(parent, partner)
            else:
                child_added = self.replicate(parent)

            if child_added:
                children_created += 1

        stats = GenerationStats(
            generation=self.generation,
            population_size=len(self.population),
            best_fitness=self.best_fitness,
            mean_fitness=self.mean_fitness,
            diversity=population_diversity(self.population),
            extinctions=self.extinction_detector.extinction_count if self.industrial_mode else 0,
        )
        if self.industrial_mode:
            self.stats_tracker.record(stats)

        return {
            "generation": self.generation,
            "population_size": len(self.population),
            "killed": killed,
            "children": children_created,
            "best_fitness": self.best_fitness,
            "mean_fitness": self.mean_fitness,
            "diversity": stats.diversity,
            "extinctions": stats.extinctions,
        }

    @property
    def best_organism(self) -> Optional[Organism]:
        """Return the highest-fitness living organism, if one is evaluated."""
        alive_with_fitness = [o for o in self.population if o.alive and o.fitness]
        return (
            max(
                alive_with_fitness,
                key=lambda o: o.fitness.composite if o.fitness is not None else float("-inf"),
            )
            if alive_with_fitness
            else None
        )

    @property
    def best_fitness(self) -> float:
        """Return the best living composite fitness, or zero when unavailable."""
        b = self.best_organism
        return b.fitness.composite if b and b.fitness else 0.0

    @property
    def mean_fitness(self) -> float:
        """Return mean composite fitness across evaluated organisms."""
        fits = [o.fitness.composite for o in self.population if o.fitness]
        return float(np.mean(fits)) if fits else 0.0


__all__ = ["ReplicationEngine"]
