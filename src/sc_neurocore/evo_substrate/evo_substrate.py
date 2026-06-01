# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Self-Replicating / Evolutionary SC Substrate

"""Open-ended evolution of SC neuromorphic networks at hardware speed.

Networks can emit mutated child networks (as NIR or Verilog) that run
on separate FPGA tiles and compete/evolve under a fitness function.

Architecture:

    ┌────────────────────────────────────────────┐
    │             EvolutionArena                  │
    │  ┌──────────┐ ┌──────────┐ ┌──────────┐   │
    │  │Organism 0│ │Organism 1│ │Organism N│   │
    │  │(parent)  │ │(child)   │ │(mutant)  │   │
    │  │  Genome  │ │  Genome  │ │  Genome  │   │
    │  │  ├ topo  │ │  ├ topo  │ │  ├ topo  │   │
    │  │  ├ neuro │ │  ├ neuro │ │  ├ neuro │   │
    │  │  └ plast │ │  └ plast │ │  └ plast │   │
    │  └────┬─────┘ └────┬─────┘ └────┬─────┘   │
    │       │ fitness     │ fitness    │ fitness  │
    │  ┌────▼─────────────▼───────────▼─────┐    │
    │  │         FitnessEvaluator            │    │
    │  └────────────────┬───────────────────┘    │
    │                   │ selection               │
    │  ┌────────────────▼───────────────────┐    │
    │  │      ReplicationEngine             │    │
    │  │  replicate() → mutate() → deploy() │    │
    │  └────────────────────────────────────┘    │
    └────────────────────────────────────────────┘

Compatible with:
- ``neurons/models/arcane_neuron.py`` — base neuron model
- ``meta_plasticity/`` — mutable plasticity rules
- ``hdl_gen/verilog_generator.py`` — HDL emission target
- ``hypervisor/`` — FPGA tile isolation for organisms
- ``safety_cert/`` — safety constraints on mutation space
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import textwrap
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from sc_neurocore.fault_injection import (
    DegradationAction,
    DegradationPlan,
    FaultModel,
    GracefulDegradationPolicy,
)

_ec: Optional[Any]
try:
    _ec = importlib.import_module("sc_neurocore.evo_substrate.evo_substrate_core")
    _HAS_RUST_EVO = True
except ImportError:
    _ec = None
    _HAS_RUST_EVO = False


# ── Genome ───────────────────────────────────────────────────────────


@dataclass
class TopologyGene:
    """Encodes the network topology."""

    num_neurons: int = 16
    num_layers: int = 2
    connectivity: float = 0.3  # connection probability
    recurrent_fraction: float = 0.1
    bitstream_length: int = 256

    def to_vector(self) -> np.ndarray:
        return np.array(
            [
                self.num_neurons,
                self.num_layers,
                self.connectivity,
                self.recurrent_fraction,
                self.bitstream_length,
            ]
        )

    @classmethod
    def from_vector(cls, v: np.ndarray) -> TopologyGene:
        return cls(
            num_neurons=max(2, int(v[0])),
            num_layers=max(1, int(v[1])),
            connectivity=float(np.clip(v[2], 0.01, 1.0)),
            recurrent_fraction=float(np.clip(v[3], 0.0, 0.5)),
            bitstream_length=max(32, int(v[4])),
        )


@dataclass
class NeuronGene:
    """Encodes neuron-level parameters (ArcaneNeuron-compatible)."""

    tau_fast: float = 5.0
    tau_work: float = 200.0
    tau_deep: float = 10000.0
    theta: float = 1.0
    gamma: float = 0.2
    delta_conf: float = 0.3
    kappa: float = 5.0
    w_inh: float = 0.3

    def to_vector(self) -> np.ndarray:
        return np.array(
            [
                self.tau_fast,
                self.tau_work,
                self.tau_deep,
                self.theta,
                self.gamma,
                self.delta_conf,
                self.kappa,
                self.w_inh,
            ]
        )

    @classmethod
    def from_vector(cls, v: np.ndarray) -> NeuronGene:
        return cls(
            tau_fast=max(0.5, float(v[0])),
            tau_work=max(1.0, float(v[1])),
            tau_deep=max(10.0, float(v[2])),
            theta=max(0.1, float(v[3])),
            gamma=float(np.clip(v[4], 0.0, 1.0)),
            delta_conf=float(np.clip(v[5], 0.0, 1.0)),
            kappa=max(0.1, float(v[6])),
            w_inh=float(np.clip(v[7], 0.0, 1.0)),
        )


@dataclass
class PlasticityGene:
    """Encodes plasticity rule parameters."""

    stdp_lr: float = 0.01
    stdp_tau_plus: float = 20.0
    stdp_tau_minus: float = 20.0
    stp_u_base: float = 0.5
    homeostatic_rate: float = 0.001
    meta_sensitivity: float = 1.0

    def to_vector(self) -> np.ndarray:
        return np.array(
            [
                self.stdp_lr,
                self.stdp_tau_plus,
                self.stdp_tau_minus,
                self.stp_u_base,
                self.homeostatic_rate,
                self.meta_sensitivity,
            ]
        )

    @classmethod
    def from_vector(cls, v: np.ndarray) -> PlasticityGene:
        return cls(
            stdp_lr=max(1e-6, float(v[0])),
            stdp_tau_plus=max(1.0, float(v[1])),
            stdp_tau_minus=max(1.0, float(v[2])),
            stp_u_base=float(np.clip(v[3], 0.01, 0.99)),
            homeostatic_rate=max(1e-6, float(v[4])),
            meta_sensitivity=max(0.1, float(v[5])),
        )


@dataclass
class Genome:
    """Complete genome for an evolving SC organism."""

    genome_id: str = ""
    parent_id: str = ""
    generation: int = 0
    topology: TopologyGene = field(default_factory=TopologyGene)
    neuron: NeuronGene = field(default_factory=NeuronGene)
    plasticity: PlasticityGene = field(default_factory=PlasticityGene)
    weight_seed: int = 42
    identity_deep: float = 0.0

    def to_vector(self) -> np.ndarray:
        return np.concatenate(
            [
                self.topology.to_vector(),
                self.neuron.to_vector(),
                self.plasticity.to_vector(),
            ]
        )

    @classmethod
    def from_vector(cls, v: np.ndarray, gen: int = 0) -> Genome:
        return cls(
            generation=gen,
            topology=TopologyGene.from_vector(v[0:5]),
            neuron=NeuronGene.from_vector(v[5:13]),
            plasticity=PlasticityGene.from_vector(v[13:19]),
        )

    @property
    def vector_dim(self) -> int:
        return len(self.to_vector())

    def compute_id(self) -> str:
        h = hashlib.sha256(self.to_vector().tobytes())
        self.genome_id = h.hexdigest()[:12]
        return self.genome_id


# ── Mutation Operators ───────────────────────────────────────────────


class MutationType(Enum):
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


# ── Genomic Distance / Speciation ────────────────────────────────────


def genomic_distance(a: Genome, b: Genome) -> float:
    """Normalised L1 distance between genome vectors.

    Dispatches to the Rust ``evo_substrate_core.py_genomic_distance`` when
    the compiled extension is importable. The NumPy fallback is kept as
    the reference implementation and produces bit-exact identical values.
    """
    va, vb = a.to_vector(), b.to_vector()
    if _HAS_RUST_EVO and _ec is not None:
        return float(
            _ec.py_genomic_distance(
                np.ascontiguousarray(va, dtype=np.float64),
                np.ascontiguousarray(vb, dtype=np.float64),
            )
        )
    diffs = va - vb
    norms = np.abs(va) + np.abs(vb) + 1e-10
    return float(np.mean(np.abs(diffs) / norms))


def assign_species(
    population: List[Organism],
    threshold: float = 0.3,
) -> Dict[int, List[Organism]]:
    """Assign organisms to species by genomic distance.

    First organism of each species is the representative.
    """
    species: Dict[int, List[Organism]] = {}
    representatives: Dict[int, Genome] = {}
    next_id = 0

    for org in population:
        placed = False
        for sid, rep in representatives.items():
            if genomic_distance(org.genome, rep) < threshold:
                species[sid].append(org)
                placed = True
                break
        if not placed:
            species[next_id] = [org]
            representatives[next_id] = org.genome
            next_id += 1

    return species


# ── Diversity Metric ─────────────────────────────────────────────────


def population_diversity(population: List[Organism]) -> float:
    """Mean pairwise genomic distance (0 = clones, 1 = max diversity)."""
    if len(population) < 2:
        return 0.0
    dists = []
    for i in range(len(population)):
        for j in range(i + 1, len(population)):
            dists.append(genomic_distance(population[i].genome, population[j].genome))
    return float(np.mean(dists))


# ── Lineage Tracker ──────────────────────────────────────────────────


@dataclass
class LineageRecord:
    """One entry in the ancestry log."""

    genome_id: str
    parent_id: str
    generation: int
    mutation_type: str
    fitness: float = 0.0


class LineageTracker:
    """Tracks ancestry graph for all organisms."""

    def __init__(self) -> None:
        self.records: List[LineageRecord] = []
        self._by_id: Dict[str, LineageRecord] = {}

    def record(self, organism: Organism, mutation_type: str = "seed") -> None:
        fit = organism.fitness.composite if organism.fitness else 0.0
        rec = LineageRecord(
            genome_id=organism.genome.genome_id,
            parent_id=organism.genome.parent_id,
            generation=organism.genome.generation,
            mutation_type=mutation_type,
            fitness=fit,
        )
        self.records.append(rec)
        self._by_id[rec.genome_id] = rec

    def get_ancestors(self, genome_id: str) -> List[LineageRecord]:
        """Walk the ancestry chain to the root."""
        chain = []
        current = genome_id
        while current in self._by_id:
            rec = self._by_id[current]
            chain.append(rec)
            current = rec.parent_id
        return chain

    @property
    def num_records(self) -> int:
        return len(self.records)


# ── Fitness Evaluation ──────────────────────────────────────────────


class FitnessType(Enum):
    ACCURACY = "accuracy"
    ENERGY = "energy"
    LATENCY = "latency"
    COMPOSITE = "composite"


@dataclass
class FitnessResult:
    """Fitness evaluation result."""

    genome_id: str
    accuracy: float = 0.0
    energy_score: float = 0.0
    latency_score: float = 0.0
    composite: float = 0.0

    def compute_composite(
        self, w_acc: float = 0.5, w_energy: float = 0.3, w_latency: float = 0.2
    ) -> float:
        self.composite = (
            w_acc * self.accuracy + w_energy * self.energy_score + w_latency * self.latency_score
        )
        return self.composite


class FitnessEvaluator:
    """Evaluates organism fitness from simulation metrics."""

    def __init__(self, fitness_type: FitnessType = FitnessType.COMPOSITE) -> None:
        self.fitness_type = fitness_type

    def evaluate(self, genome: Genome, metrics: Dict[str, float]) -> FitnessResult:
        result = FitnessResult(genome_id=genome.genome_id)
        result.accuracy = metrics.get("accuracy", 0.0)

        # Energy: fewer neurons + shorter bitstreams = better
        neuron_pen = min(genome.topology.num_neurons / 1024.0, 1.0)
        bs_pen = min(genome.topology.bitstream_length / 1024.0, 1.0)
        result.energy_score = max(0.0, 1.0 - 0.5 * neuron_pen - 0.5 * bs_pen)

        # Latency: fewer layers = faster
        result.latency_score = max(0.0, 1.0 - genome.topology.num_layers / 10.0)

        result.compute_composite()
        return result


# ── Replication Engine ───────────────────────────────────────────────


@dataclass
class Organism:
    """One evolving SC organism."""

    genome: Genome
    fitness: Optional[FitnessResult] = None
    alive: bool = True
    tile_id: Optional[int] = None
    birth_generation: int = 0
    lifespan_steps: int = 0
    runtime_fault_checks: List[RuntimeFaultCheck] = field(default_factory=list)


@dataclass(frozen=True)
class RuntimeFaultConfig:
    """Runtime fault-check settings for evolved SC organisms."""

    fault_model: FaultModel = FaultModel.BIT_FLIP
    ber: float = 0.0
    seed_offset: int = 0
    sample_neurons: int = 8
    fitness_penalty_on_extend: float = 0.95
    fitness_penalty_on_replay: float = 0.85


@dataclass(frozen=True)
class RuntimeFaultCheck:
    """Recorded runtime fault/degradation decision for one organism."""

    organism_id: str
    generation: int
    action: str
    recommended_bitstream_length: int
    replay_seed: int
    affected_ratio: float
    audit_status: str
    reason: str

    @classmethod
    def from_plan(cls, organism: Organism, plan: DegradationPlan) -> RuntimeFaultCheck:
        return cls(
            organism_id=organism.genome.genome_id,
            generation=organism.genome.generation,
            action=plan.action.value,
            recommended_bitstream_length=plan.recommended_bitstream_length,
            replay_seed=plan.replay_seed,
            affected_ratio=plan.observation.affected_ratio,
            audit_status=plan.observation.audit.status.value,
            reason=plan.reason,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-ready fault-check summary."""
        return {
            "organism_id": self.organism_id,
            "generation": self.generation,
            "action": self.action,
            "recommended_bitstream_length": self.recommended_bitstream_length,
            "replay_seed": self.replay_seed,
            "affected_ratio": self.affected_ratio,
            "audit_status": self.audit_status,
            "reason": self.reason,
        }


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
                    if org.fitness:
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
        b = self.best_organism
        return b.fitness.composite if b and b.fitness else 0.0

    @property
    def mean_fitness(self) -> float:
        fits = [o.fitness.composite for o in self.population if o.fitness]
        return float(np.mean(fits)) if fits else 0.0


# ── NIR / Verilog Emission ──────────────────────────────────────────


class OrganismEmitter:
    """Emits evolved organisms as NIR graph or Verilog."""

    @staticmethod
    def to_nir(genome: Genome) -> Dict[str, Any]:
        """Emit a simplified NIR-compatible graph dict."""
        nodes = {}
        for i in range(genome.topology.num_neurons):
            nodes[f"n{i}"] = {
                "type": "ArcaneNeuron",
                "tau_fast": genome.neuron.tau_fast,
                "tau_work": genome.neuron.tau_work,
                "tau_deep": genome.neuron.tau_deep,
                "theta": genome.neuron.theta,
                "gamma": genome.neuron.gamma,
                "delta_conf": genome.neuron.delta_conf,
                "kappa": genome.neuron.kappa,
                "w_inh": genome.neuron.w_inh,
            }
        edges = []
        rng = np.random.default_rng(genome.weight_seed)
        for i in range(genome.topology.num_neurons):
            for j in range(genome.topology.num_neurons):
                if i != j and rng.random() < genome.topology.connectivity:
                    edges.append(
                        {"from": f"n{i}", "to": f"n{j}", "weight_q88": int(rng.integers(0, 256))}
                    )
        return {
            "genome_id": genome.genome_id,
            "generation": genome.generation,
            "nodes": nodes,
            "edges": edges,
            "bitstream_length": genome.topology.bitstream_length,
        }

    @staticmethod
    def to_verilog(genome: Genome, module_name: Optional[str] = None) -> str:
        """Emit Verilog wrapper for the organism."""
        name = module_name or f"sc_organism_{genome.genome_id[:8]}"
        n = genome.topology.num_neurons
        bs = genome.topology.bitstream_length
        return textwrap.dedent(f"""\
// SC-NeuroCore — Evolved Organism: {genome.genome_id}
// Generation: {genome.generation} | Neurons: {n} | Bitstream: {bs}

module {name} #(
    parameter NUM_NEURONS = {n},
    parameter BITSTREAM_W = {bs},
    parameter TAU_FAST    = {int(genome.neuron.tau_fast)},
    parameter TAU_WORK    = {int(genome.neuron.tau_work)},
    parameter THETA_Q88   = {int(genome.neuron.theta * 256)}
)(
    input  wire                    clk,
    input  wire                    rst_n,
    input  wire [BITSTREAM_W-1:0]  sc_input  [0:NUM_NEURONS-1],
    output wire [BITSTREAM_W-1:0]  sc_output [0:NUM_NEURONS-1],
    output wire [NUM_NEURONS-1:0]  spike_out
);

    genvar i;
    generate
        for (i = 0; i < NUM_NEURONS; i = i + 1) begin : neuron_gen
            sc_lif_neuron #(
                .BITSTREAM_W(BITSTREAM_W),
                .THRESHOLD(THETA_Q88)
            ) u_neuron (
                .clk(clk),
                .rst_n(rst_n),
                .bitstream_in(sc_input[i]),
                .bitstream_out(sc_output[i]),
                .spike(spike_out[i])
            );
        end
    endgenerate

endmodule
""")

    @staticmethod
    def to_photonic_netlist(genome: Genome, pml_layers: int = 12) -> Dict[str, Any]:
        """Emit a photonic netlist compatible with the optics PhotonicCompiler."""
        return {
            "version": "1.0",
            "metadata": {
                "genome_id": genome.genome_id,
                "generation": genome.generation,
                "num_neurons": genome.topology.num_neurons,
            },
            "parameters": {
                "wavelength": 1.55e-6,
                "n_core": 3.48,
                "n_clad": 1.44,
                "pml_layers": pml_layers,
            },
            "waveguides": [
                {"id": f"wg_{i}", "width": 0.5, "length": 10.0}
                for i in range(genome.topology.num_neurons)
            ],
        }


# ── Mutation Safety Bounds ──────────────────────────────────


@dataclass
class SafetyBounds:
    """Constrains the mutation space to prevent runaway replication.

    Enforces hard limits on genome parameters that could cause
    resource exhaustion or unsafe behaviour on FPGA tiles.
    """

    max_neurons: int = 1024
    min_neurons: int = 4
    max_layers: int = 16
    max_bitstream: int = 4096
    min_bitstream: int = 32
    max_connectivity: float = 1.0
    max_tau_deep: float = 100000.0
    max_replications_per_gen: int = 64

    def clamp(self, genome: Genome) -> Genome:
        genome.topology.num_neurons = int(
            np.clip(genome.topology.num_neurons, self.min_neurons, self.max_neurons)
        )
        genome.topology.num_layers = min(self.max_layers, max(1, genome.topology.num_layers))
        genome.topology.bitstream_length = int(
            np.clip(genome.topology.bitstream_length, self.min_bitstream, self.max_bitstream)
        )
        genome.topology.connectivity = float(
            np.clip(genome.topology.connectivity, 0.01, self.max_connectivity)
        )
        genome.neuron.tau_deep = min(self.max_tau_deep, genome.neuron.tau_deep)
        return genome

    def is_within_bounds(self, genome: Genome) -> bool:
        return (
            self.min_neurons <= genome.topology.num_neurons <= self.max_neurons
            and 1 <= genome.topology.num_layers <= self.max_layers
            and self.min_bitstream <= genome.topology.bitstream_length <= self.max_bitstream
        )


# ── FPGA Tile Deployment Tracker ────────────────────────────


@dataclass
class TileAllocation:
    """Maps an organism to a physical FPGA tile."""

    organism_id: str
    tile_id: int
    partition_id: int = 0
    deployed: bool = False
    bitstream_hash: str = ""


class TileDeploymentTracker:
    """Tracks which organisms are deployed on which FPGA tiles."""

    def __init__(self, num_tiles: int = 8) -> None:
        self.num_tiles = num_tiles
        self.allocations: Dict[int, Optional[TileAllocation]] = {i: None for i in range(num_tiles)}

    def deploy(self, organism: Organism, tile_id: int) -> TileAllocation:
        alloc = TileAllocation(
            organism_id=organism.genome.genome_id,
            tile_id=tile_id,
            deployed=True,
            bitstream_hash=organism.genome.genome_id,
        )
        self.allocations[tile_id] = alloc
        organism.tile_id = tile_id
        return alloc

    def evict(self, tile_id: int) -> None:
        self.allocations[tile_id] = None

    @property
    def free_tiles(self) -> List[int]:
        return [tid for tid, a in self.allocations.items() if a is None]

    @property
    def utilisation(self) -> float:
        used = sum(1 for a in self.allocations.values() if a is not None)
        return used / self.num_tiles if self.num_tiles > 0 else 0.0


# ── Fitness Hall of Fame ────────────────────────────────────


class HallOfFame:
    """Maintains the top-N organisms across all generations."""

    def __init__(self, max_size: int = 10) -> None:
        self.max_size = max_size
        self.entries: List[Tuple[float, Genome]] = []  # (fitness, genome)

    def update(self, organism: Organism) -> bool:
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
        return self.entries[0][0] if self.entries else 0.0

    @property
    def size(self) -> int:
        return len(self.entries)


# ── Island Model ────────────────────────────────────────────


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
        return sum(len(isl.population) for isl in self.islands.values())


# ── Genome Serialization ────────────────────────────────────


class GenomeSerializer:
    """Serializes/deserializes genomes for persistence."""

    @staticmethod
    def to_dict(genome: Genome) -> Dict[str, Any]:
        return {
            "genome_id": genome.genome_id,
            "parent_id": genome.parent_id,
            "generation": genome.generation,
            "weight_seed": genome.weight_seed,
            "identity_deep": genome.identity_deep,
            "vector": genome.to_vector().tolist(),
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> Genome:
        v = np.array(d["vector"])
        g = Genome.from_vector(v, d.get("generation", 0))
        g.genome_id = d.get("genome_id", "")
        g.parent_id = d.get("parent_id", "")
        g.weight_seed = d.get("weight_seed", 42)
        g.identity_deep = d.get("identity_deep", 0.0)
        return g


# ── Novelty Search ──────────────────────────────────────────


class NoveltyArchive:
    """Behavioural novelty archive for novelty search."""

    def __init__(self, k_nearest: int = 5, threshold: float = 0.1) -> None:
        self.k_nearest = k_nearest
        self.threshold = threshold
        self.archive: List[np.ndarray] = []

    def novelty_score(self, behaviour: np.ndarray) -> float:
        if not self.archive:
            return 1.0
        dists = [float(np.linalg.norm(behaviour - a)) for a in self.archive]
        dists.sort()
        k = min(self.k_nearest, len(dists))
        return float(np.mean(dists[:k]))

    def maybe_add(self, behaviour: np.ndarray) -> bool:
        score = self.novelty_score(behaviour)
        if score > self.threshold:
            self.archive.append(behaviour.copy())
            return True
        return False

    @property
    def size(self) -> int:
        return len(self.archive)


# ── Resource Budget ─────────────────────────────────────────


@dataclass
class ResourceBudget:
    """Per-organism resource constraints."""

    max_area_um2: float = 1e6
    max_power_mw: float = 100.0
    max_neurons: int = 1024

    def check(self, genome: Genome) -> Tuple[bool, List[str]]:
        violations = []
        if genome.topology.num_neurons > self.max_neurons:
            violations.append(f"neurons={genome.topology.num_neurons}>{self.max_neurons}")
        est_area = genome.topology.num_neurons * genome.topology.bitstream_length * 0.1
        if est_area > self.max_area_um2:
            violations.append(f"area={est_area:.0f}>{self.max_area_um2:.0f}")
        return (len(violations) == 0, violations)


# ── Extinction Events ───────────────────────────────────────


class ExtinctionDetector:
    """Detects population stagnation and triggers extinction events."""

    def __init__(self, stagnation_gens: int = 10, kill_fraction: float = 0.9) -> None:
        self.stagnation_gens = stagnation_gens
        self.kill_fraction = kill_fraction
        self._best_history: List[float] = []
        self.extinction_count: int = 0

    def check(self, best_fitness: float) -> bool:
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
        self.predators.append(CoevoOrganism(organism, CoevoRole.PREDATOR))

    def add_prey(self, organism: Organism) -> None:
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
        return len(self.predators) + len(self.prey)


# ── Formal Safety Guard ────────────────────────────────────


@dataclass
class SafetyCheckResult:
    """Result of a formal safety check on emitted Verilog/NIR."""

    genome_id: str
    passed: bool
    violations: List[str] = field(default_factory=list)
    neuron_count_ok: bool = True
    connectivity_ok: bool = True
    bitstream_ok: bool = True


class FormalSafetyGuard:
    """Validates emitted organisms against safety constraints before deployment.

    Links to the safety_cert module for IEC 61508 compliance.
    """

    def __init__(self, bounds: Optional[SafetyBounds] = None) -> None:
        self.bounds = bounds or SafetyBounds()
        self.checked: int = 0
        self.rejected: int = 0

    def check(self, genome: Genome) -> SafetyCheckResult:
        self.checked += 1
        violations = []
        n_ok = genome.topology.num_neurons <= self.bounds.max_neurons
        c_ok = genome.topology.connectivity <= self.bounds.max_connectivity
        b_ok = genome.topology.bitstream_length <= self.bounds.max_bitstream

        if not n_ok:
            violations.append(f"neurons={genome.topology.num_neurons}>{self.bounds.max_neurons}")
        if not c_ok:
            violations.append(
                f"connectivity={genome.topology.connectivity}>{self.bounds.max_connectivity}"
            )
        if not b_ok:
            violations.append(
                f"bitstream={genome.topology.bitstream_length}>{self.bounds.max_bitstream}"
            )

        passed = len(violations) == 0
        if not passed:
            self.rejected += 1

        return SafetyCheckResult(
            genome_id=genome.genome_id,
            passed=passed,
            violations=violations,
            neuron_count_ok=n_ok,
            connectivity_ok=c_ok,
            bitstream_ok=b_ok,
        )

    @property
    def rejection_rate(self) -> float:
        return self.rejected / self.checked if self.checked > 0 else 0.0


# ── Tournament Selection ───────────────────────────────────


class TournamentSelector:
    """Tournament selection with configurable pressure."""

    def __init__(self, tournament_size: int = 3) -> None:
        self.tournament_size = tournament_size

    def select(self, population: List[Organism], rng: np.random.Generator) -> Organism:
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
        return [self.select(population, rng) for _ in range(n)]


# ── Multi-Objective Pareto Front ───────────────────────────


def dominates(a: FitnessResult, b: FitnessResult) -> bool:
    """True if a Pareto-dominates b."""
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
        return len(self.front)


# ── Age-Based Regulation ───────────────────────────────────


class AgeRegulator:
    """Culls organisms that exceed a maximum lifespan."""

    def __init__(self, max_age: int = 20) -> None:
        self.max_age = max_age

    def apply(self, population: List[Organism], current_generation: int) -> int:
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
        bm = compute_bloat(genome)
        if bm.bloat_score > self.threshold:
            excess = bm.bloat_score - self.threshold
            return fitness * max(0.1, 1.0 - self.penalty_weight * excess)
        return fitness


# ── Fitness Sharing ────────────────────────────────────────


def shared_fitness(
    organism: Organism,
    population: List[Organism],
    sigma: float = 0.3,
) -> float:
    """Shared fitness: divide by niche count to prevent species domination."""
    if organism.fitness is None:
        return 0.0
    raw = organism.fitness.composite
    niche_count = sum(
        1.0 for other in population if genomic_distance(organism.genome, other.genome) < sigma
    )
    return raw / max(1.0, niche_count)


# ── Developmental Encoding / CPPN ──────────────────────────


class ActivationFunc(Enum):
    SIN = "sin"
    GAUSS = "gauss"
    LINEAR = "linear"
    SIGMOID = "sigmoid"
    STEP = "step"


@dataclass
class CPPNNode:
    """One node in a CPPN network."""

    node_id: int
    activation: ActivationFunc = ActivationFunc.LINEAR
    bias: float = 0.0


@dataclass
class CPPNEdge:
    """One edge in a CPPN network."""

    src: int
    dst: int
    weight: float = 1.0
    enabled: bool = True


class CPPNGenome:
    """Compositional Pattern Producing Network for developmental encoding."""

    def __init__(self) -> None:
        self.nodes: List[CPPNNode] = [
            CPPNNode(0, ActivationFunc.LINEAR),  # input x
            CPPNNode(1, ActivationFunc.LINEAR),  # input y
            CPPNNode(2, ActivationFunc.SIGMOID),  # output
        ]
        self.edges: List[CPPNEdge] = [
            CPPNEdge(0, 2, 1.0),
            CPPNEdge(1, 2, 1.0),
        ]

    def query(self, x: float, y: float) -> float:
        """Query the CPPN at coordinates (x, y)."""
        values = {0: x, 1: y}
        for node in self.nodes[2:]:
            total = node.bias
            for edge in self.edges:
                if edge.dst == node.node_id and edge.enabled and edge.src in values:
                    total += edge.weight * values[edge.src]
            values[node.node_id] = self._activate(total, node.activation)
        return values.get(2, 0.0)

    @staticmethod
    def _activate(x: float, func: ActivationFunc) -> float:
        if func == ActivationFunc.SIN:
            return float(np.sin(x))
        if func == ActivationFunc.GAUSS:
            return float(np.exp(-x * x))
        if func == ActivationFunc.SIGMOID:
            return float(1.0 / (1.0 + np.exp(-np.clip(x, -10, 10))))
        if func == ActivationFunc.STEP:
            return 1.0 if x > 0 else 0.0
        return float(x)  # LINEAR

    def generate_weight_matrix(self, rows: int, cols: int) -> np.ndarray:
        """Generate a weight matrix by querying CPPN at grid positions."""
        w = np.zeros((rows, cols))
        for r in range(rows):
            for c in range(cols):
                x = 2.0 * r / max(1, rows - 1) - 1.0
                y = 2.0 * c / max(1, cols - 1) - 1.0
                w[r, c] = self.query(x, y)
        return w

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_edges(self) -> int:
        return len(self.edges)


# ── HW-in-the-Loop Fitness Feedback ────────────────────────


@dataclass
class HWFitnessReport:
    """Fitness feedback from actual FPGA execution."""

    genome_id: str
    fpga_cycles: int = 0
    fpga_power_mw: float = 0.0
    fpga_accuracy: float = 0.0
    fmax_mhz: float = 0.0
    timing_met: bool = True

    @property
    def hw_composite(self) -> float:
        time_score = min(1.0, 100.0 / max(1.0, self.fmax_mhz)) if self.fmax_mhz > 0 else 0.0
        return (
            0.5 * self.fpga_accuracy
            + 0.3 * (1.0 - time_score)
            + 0.2 * (1.0 if self.timing_met else 0.0)
        )


class HWFitnessCollector:
    """Collects HW fitness from deployed organisms."""

    def __init__(self) -> None:
        self.reports: Dict[str, HWFitnessReport] = {}

    def submit(self, report: HWFitnessReport) -> None:
        self.reports[report.genome_id] = report

    def get(self, genome_id: str) -> Optional[HWFitnessReport]:
        return self.reports.get(genome_id)

    @property
    def total_reports(self) -> int:
        return len(self.reports)


# ── Evolutionary Statistics ────────────────────────────────


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
        self.history.append(stats)

    @property
    def generations_tracked(self) -> int:
        return len(self.history)

    @property
    def fitness_trajectory(self) -> List[float]:
        return [s.best_fitness for s in self.history]

    @property
    def diversity_trajectory(self) -> List[float]:
        return [s.diversity for s in self.history]

    def improvement_rate(self) -> float:
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
    return entropy + np.log2(1 + topology_complexity)


class ComplexityTracker:
    """Tracks population complexity over generations."""

    def __init__(self) -> None:
        self.history: List[Tuple[int, float, float]] = []  # (gen, mean, max)

    def record(self, generation: int, population: List[Organism]) -> None:
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
        return [h[1] for h in self.history]

    @property
    def is_complexifying(self) -> bool:
        if len(self.history) < 3:
            return False
        return self.history[-1][1] > self.history[0][1]
