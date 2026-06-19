# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Self-Evolving Meta-Plasticity SC Engine

"""Self-evolving meta-plasticity for lifelong/continual learning.

Extends ArcaneNeuron's self-referential model so the network dynamically
rewrites its own plasticity rules and bitstream parameters at runtime
using an internal SC "meta-controller" driven by SCPN L16 Director signals.

Architecture:

    ┌─────────────┐
    │  L16 Director│───► GCI / veto / will
    └──────┬──────┘
           │ meta-control signals (bitstream-encoded)
    ┌──────▼──────┐
    │ MetaController│─► rewrites PlasticityRuleSet
    └──────┬──────┘
           │ adapted rules
    ┌──────▼──────────────────┐
    │ PlasticityRuleSet        │
    │   ├── STDP (tau, A, lr)  │
    │   ├── STP  (U, tau_d/f)  │
    │   ├── Homeostatic        │
    │   └── Bitstream (length) │
    └──────┬──────────────────┘
           │ applied to
    ┌──────▼──────┐
    │ ArcaneNeuron │ (identity substrate)
    └─────────────┘

Key innovations:
- **PlasticityRuleSet**: Parametric STDP/STP/homeostatic rules as mutable
  dataclass. Can be serialised/deserialised for checkpointing.
- **MetaController**: SC-domain controller that observes network performance
  (surprise, novelty, GCI) and emits rule modifications as bitstreams.
- **RuleEvolver**: Evolutionary engine that maintains a population of
  candidate rule sets, selects based on fitness, and recombines.
- **MetaPlasticityEngine**: Top-level orchestrator that connects ArcaneNeuron
  populations, the meta-controller, and the rule evolver.

Compatible with:
- ``neurons/models/arcane_neuron.py`` — ArcaneNeuron model
- ``scpn/layers/l15_meta.py`` — L15 GCI meta-monitor
- ``adapters/holonomic/l16_meta.py`` — L16 Director adapter
- ``synapses/short_term_plasticity.py`` — STP synapse model
"""

from __future__ import annotations

import copy
import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

import numpy as np

try:
    from sc_neurocore._native import learning_bridge as _learning_bridge

    _HAS_RUST_LEARNING = _learning_bridge.is_available() and all(
        hasattr(_learning_bridge, _name)
        for _name in ("RustPlasticityRule", "RULE_STDP", "RULE_BCM")
    )
    del _learning_bridge
except ImportError:
    _HAS_RUST_LEARNING = False


# ── Plasticity Rule Set ──────────────────────────────────────────────


@dataclass
class STDPParams:
    """Mutable STDP parameters."""

    tau_plus: float = 20.0  # ms
    tau_minus: float = 20.0  # ms
    a_plus: float = 0.01
    a_minus: float = 0.012
    lr: float = 0.01

    def to_vector(self) -> np.ndarray[Any, Any]:
        return np.array([self.tau_plus, self.tau_minus, self.a_plus, self.a_minus, self.lr])

    @classmethod
    def from_vector(cls, v: np.ndarray[Any, Any]) -> STDPParams:
        return cls(
            tau_plus=max(1.0, float(v[0])),
            tau_minus=max(1.0, float(v[1])),
            a_plus=max(1e-6, float(v[2])),
            a_minus=max(1e-6, float(v[3])),
            lr=max(1e-6, float(v[4])),
        )


@dataclass
class STPParams:
    """Mutable short-term plasticity parameters."""

    u_base: float = 0.5
    tau_d: float = 200.0  # ms, depression
    tau_f: float = 20.0  # ms, facilitation

    def to_vector(self) -> np.ndarray[Any, Any]:
        return np.array([self.u_base, self.tau_d, self.tau_f])

    @classmethod
    def from_vector(cls, v: np.ndarray[Any, Any]) -> STPParams:
        return cls(
            u_base=float(np.clip(v[0], 0.01, 0.99)),
            tau_d=max(1.0, float(v[1])),
            tau_f=max(1.0, float(v[2])),
        )


@dataclass
class HomeostaticParams:
    """Homeostatic plasticity: target rate + gain modulation."""

    target_rate_hz: float = 5.0
    gain_adaptation_rate: float = 0.001
    current_gain: float = 1.0

    def adapt(self, measured_rate_hz: float) -> float:
        """Adjust gain to push firing rate toward target."""
        error = self.target_rate_hz - measured_rate_hz
        self.current_gain += self.gain_adaptation_rate * error
        self.current_gain = max(0.1, min(10.0, self.current_gain))
        return self.current_gain


@dataclass
class BitstreamParams:
    """Mutable SC bitstream parameters."""

    length: int = 256
    lfsr_seed: int = 0xACE1
    precision_bits: int = 8  # Q8.8


@dataclass
class PlasticityRuleSet:
    """Complete set of mutable plasticity rules.

    This is the "genome" that the meta-controller evolves.
    """

    stdp: STDPParams = field(default_factory=STDPParams)
    stp: STPParams = field(default_factory=STPParams)
    homeostatic: HomeostaticParams = field(default_factory=HomeostaticParams)
    bitstream: BitstreamParams = field(default_factory=BitstreamParams)
    generation: int = 0
    fitness: float = 0.0

    def to_vector(self) -> np.ndarray[Any, Any]:
        """Serialise all mutable params to a flat vector."""
        return np.concatenate(
            [
                self.stdp.to_vector(),
                self.stp.to_vector(),
                np.array(
                    [
                        self.homeostatic.target_rate_hz,
                        self.homeostatic.gain_adaptation_rate,
                        float(self.bitstream.length),
                    ]
                ),
            ]
        )

    @classmethod
    def from_vector(cls, v: np.ndarray[Any, Any], gen: int = 0) -> PlasticityRuleSet:
        stdp = STDPParams.from_vector(v[0:5])
        stp = STPParams.from_vector(v[5:8])
        homeo = HomeostaticParams(
            target_rate_hz=max(0.1, float(v[8])),
            gain_adaptation_rate=max(1e-6, float(v[9])),
        )
        bs = BitstreamParams(length=max(32, int(v[10])))
        return cls(stdp=stdp, stp=stp, homeostatic=homeo, bitstream=bs, generation=gen)

    @property
    def vector_dim(self) -> int:
        return len(self.to_vector())

    def copy(self) -> PlasticityRuleSet:
        return copy.deepcopy(self)


# ── Meta-Controller ──────────────────────────────────────────────────


class MetaSignalType(Enum):
    """Types of meta-control signals from L16."""

    INCREASE_LR = "increase_lr"
    DECREASE_LR = "decrease_lr"
    WIDEN_WINDOW = "widen_window"
    NARROW_WINDOW = "narrow_window"
    INCREASE_HOMEOSTATIC = "increase_homeostatic"
    RESET_STP = "reset_stp"
    NO_OP = "no_op"


@dataclass
class MetaControlSignal:
    """One meta-control directive."""

    signal_type: MetaSignalType
    magnitude: float = 0.1
    target_param: str = ""


class MetaController:
    """SC-domain meta-controller that rewrites plasticity rules.

    Observes network performance metrics (surprise, novelty, GCI,
    firing rates) and emits rule modifications. The controller itself
    uses SC bitstream logic for decision-making.
    """

    def __init__(self, sensitivity: float = 1.0, rng_seed: int = 42) -> None:
        self.sensitivity = sensitivity
        self.rng = np.random.default_rng(rng_seed)
        self.observation_window: Deque[Dict[str, float]] = deque(maxlen=100)
        self.signal_history: List[MetaControlSignal] = []

    def observe(self, metrics: Dict[str, float]) -> None:
        """Record one observation of network state."""
        self.observation_window.append(metrics)

    def decide(self) -> List[MetaControlSignal]:
        """Decide what meta-control signals to emit.

        Decision logic based on novelty/surprise trends:
        - High sustained novelty → increase learning rate
        - Low novelty + low surprise → decrease learning rate (consolidate)
        - High GCI variance → widen STDP window (explore)
        - Stable GCI → narrow STDP window (exploit)
        - Rate drift → increase homeostatic gain
        """
        if len(self.observation_window) < 5:
            return [MetaControlSignal(MetaSignalType.NO_OP)]

        recent = list(self.observation_window)[-10:]
        novelties = [m.get("novelty", 0.5) for m in recent]
        surprises = [m.get("surprise", 0.0) for m in recent]
        gcis = [m.get("gci", 0.5) for m in recent]

        mean_novelty = float(np.mean(novelties))
        mean_surprise = float(np.mean(surprises))
        gci_std = float(np.std(gcis))
        mean_gci = float(np.mean(gcis))

        signals = []

        # High novelty → learn faster
        if mean_novelty > 0.7 * self.sensitivity:
            signals.append(
                MetaControlSignal(
                    MetaSignalType.INCREASE_LR,
                    magnitude=0.1 * mean_novelty,
                    target_param="stdp.lr",
                )
            )
        # Low novelty + low surprise → consolidate
        elif mean_novelty < 0.3 and mean_surprise < 0.1:
            signals.append(
                MetaControlSignal(
                    MetaSignalType.DECREASE_LR,
                    magnitude=0.05,
                    target_param="stdp.lr",
                )
            )

        # Unstable GCI → widen STDP window
        if gci_std > 0.1 * self.sensitivity:
            signals.append(
                MetaControlSignal(
                    MetaSignalType.WIDEN_WINDOW,
                    magnitude=2.0,
                    target_param="stdp.tau_plus",
                )
            )
        # Stable GCI → narrow window (exploit)
        elif gci_std < 0.02 and mean_gci > 0.7:
            signals.append(
                MetaControlSignal(
                    MetaSignalType.NARROW_WINDOW,
                    magnitude=1.0,
                    target_param="stdp.tau_plus",
                )
            )

        if not signals:
            signals.append(MetaControlSignal(MetaSignalType.NO_OP))

        self.signal_history.extend(signals)
        return signals

    def apply_signals(
        self, rules: PlasticityRuleSet, signals: List[MetaControlSignal]
    ) -> PlasticityRuleSet:
        """Apply meta-control signals to a rule set (in-place)."""
        for sig in signals:
            if sig.signal_type == MetaSignalType.INCREASE_LR:
                rules.stdp.lr *= 1.0 + sig.magnitude
                rules.stdp.lr = min(rules.stdp.lr, 0.1)
            elif sig.signal_type == MetaSignalType.DECREASE_LR:
                rules.stdp.lr *= 1.0 - sig.magnitude
                rules.stdp.lr = max(rules.stdp.lr, 1e-6)
            elif sig.signal_type == MetaSignalType.WIDEN_WINDOW:
                rules.stdp.tau_plus += sig.magnitude
                rules.stdp.tau_minus += sig.magnitude
                rules.stdp.tau_plus = min(rules.stdp.tau_plus, 100.0)
                rules.stdp.tau_minus = min(rules.stdp.tau_minus, 100.0)
            elif sig.signal_type == MetaSignalType.NARROW_WINDOW:
                rules.stdp.tau_plus = max(5.0, rules.stdp.tau_plus - sig.magnitude)
                rules.stdp.tau_minus = max(5.0, rules.stdp.tau_minus - sig.magnitude)
            elif sig.signal_type == MetaSignalType.INCREASE_HOMEOSTATIC:
                rules.homeostatic.gain_adaptation_rate *= 1.5
            elif sig.signal_type == MetaSignalType.RESET_STP:
                rules.stp = STPParams()
        return rules


# ── Rule Evolver ─────────────────────────────────────────────────────


@dataclass
class RuleEvolver:
    """Evolutionary engine for plasticity rule sets.

    Maintains a population of candidate rules, evaluates fitness,
    and evolves via selection + crossover + mutation.
    """

    population_size: int = 8
    mutation_rate: float = 0.1
    mutation_scale: float = 0.05
    elite_count: int = 2
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(42))
    population: List[PlasticityRuleSet] = field(default_factory=list)
    generation: int = 0

    def __post_init__(self) -> None:
        if not self.population:
            self.population = [PlasticityRuleSet(generation=0) for _ in range(self.population_size)]

    def evaluate_fitness(self, rules: PlasticityRuleSet, metrics: Dict[str, float]) -> float:
        """Compute fitness from network performance metrics.

        Fitness = GCI * stability - surprise_penalty - rate_deviation
        """
        gci = metrics.get("gci", 0.5)
        stability = 1.0 - metrics.get("gci_std", 0.1)
        surprise_penalty = metrics.get("mean_surprise", 0.0)
        rate_dev = abs(metrics.get("mean_rate_hz", 5.0) - rules.homeostatic.target_rate_hz)
        rate_pen = min(rate_dev / 10.0, 1.0)

        fitness = gci * max(stability, 0.0) - 0.3 * surprise_penalty - 0.2 * rate_pen
        rules.fitness = fitness
        return fitness

    def select_parents(self) -> Tuple[PlasticityRuleSet, PlasticityRuleSet]:
        """Tournament selection of two parents."""
        candidates = self.rng.choice(len(self.population), size=4, replace=False)
        sorted_c = sorted(candidates, key=lambda i: self.population[i].fitness, reverse=True)
        return self.population[sorted_c[0]], self.population[sorted_c[1]]

    def crossover(self, p1: PlasticityRuleSet, p2: PlasticityRuleSet) -> PlasticityRuleSet:
        """Uniform crossover of two rule sets."""
        v1 = p1.to_vector()
        v2 = p2.to_vector()
        mask = self.rng.random(len(v1)) < 0.5
        child_v = np.where(mask, v1, v2)
        return PlasticityRuleSet.from_vector(child_v, gen=self.generation + 1)

    def mutate(self, rules: PlasticityRuleSet) -> PlasticityRuleSet:
        """Gaussian mutation of rule parameters."""
        v = rules.to_vector()
        mask = self.rng.random(len(v)) < self.mutation_rate
        noise = self.rng.normal(0, self.mutation_scale, size=len(v))
        v[mask] += noise[mask] * np.abs(v[mask] + 1e-8)
        return PlasticityRuleSet.from_vector(v, gen=self.generation + 1)

    def evolve(self) -> List[PlasticityRuleSet]:
        """Run one generation of evolution."""
        self.generation += 1
        sorted_pop = sorted(self.population, key=lambda r: r.fitness, reverse=True)

        # Elitism
        new_pop = [r.copy() for r in sorted_pop[: self.elite_count]]

        # Fill rest with crossover + mutation
        while len(new_pop) < self.population_size:
            p1, p2 = self.select_parents()
            child = self.crossover(p1, p2)
            child = self.mutate(child)
            new_pop.append(child)

        self.population = new_pop[: self.population_size]
        return self.population

    @property
    def best(self) -> PlasticityRuleSet:
        return max(self.population, key=lambda r: r.fitness)

    @property
    def mean_fitness(self) -> float:
        return float(np.mean([r.fitness for r in self.population]))


# ── Neuromodulatory State ────────────────────────────────────────────


class NeuromodulatorType(Enum):
    DOPAMINE = "dopamine"  # reward / surprise
    SEROTONIN = "serotonin"  # mood / stability
    ACETYLCHOLINE = "acetylcholine"  # attention / precision
    NOREPINEPHRINE = "norepinephrine"  # arousal / exploration


@dataclass
class NeuromodulatorState:
    """Simulated neuromodulatory tone that modulates meta-plasticity."""

    levels: Dict[NeuromodulatorType, float] = field(
        default_factory=lambda: {
            NeuromodulatorType.DOPAMINE: 0.5,
            NeuromodulatorType.SEROTONIN: 0.5,
            NeuromodulatorType.ACETYLCHOLINE: 0.5,
            NeuromodulatorType.NOREPINEPHRINE: 0.5,
        }
    )
    decay_rate: float = 0.01

    def update(self, novelty: float, surprise: float, gci: float) -> None:
        """Update neuromodulator levels from network state."""
        self.levels[NeuromodulatorType.DOPAMINE] += 0.1 * (surprise - 0.5) - self.decay_rate
        self.levels[NeuromodulatorType.SEROTONIN] += 0.05 * (gci - 0.5) - self.decay_rate
        self.levels[NeuromodulatorType.ACETYLCHOLINE] += 0.08 * (novelty - 0.5) - self.decay_rate
        self.levels[NeuromodulatorType.NOREPINEPHRINE] += 0.06 * (surprise - 0.3) - self.decay_rate

        for nm in self.levels:
            self.levels[nm] = max(0.0, min(1.0, self.levels[nm]))

    def modulation_factor(self, param: str) -> float:
        """Get a combined modulation factor for a specific parameter type."""
        da = self.levels[NeuromodulatorType.DOPAMINE]
        ach = self.levels[NeuromodulatorType.ACETYLCHOLINE]
        ne = self.levels[NeuromodulatorType.NOREPINEPHRINE]

        if param == "lr":
            return 0.5 + da + 0.3 * ne
        elif param == "tau":
            return 0.8 + 0.4 * (1.0 - ach)
        elif param == "gain":
            return 0.5 + 0.5 * ne
        return 1.0


# ── Meta-Plasticity Engine ──────────────────────────────────────────


@dataclass
class EngineConfig:
    """Configuration for the meta-plasticity engine."""

    num_neurons: int = 16
    meta_interval: int = 50
    evolve_interval: int = 500
    enable_neuromodulation: bool = True
    enable_evolution: bool = True


@dataclass
class MetaPlasticityEngine:
    """Top-level orchestrator for self-evolving meta-plasticity.

    Connects ArcaneNeuron populations, the meta-controller, and the
    rule evolver into a single lifelong learning system.
    """

    config: EngineConfig = field(default_factory=EngineConfig)
    rules: PlasticityRuleSet = field(default_factory=PlasticityRuleSet)
    controller: MetaController = field(default_factory=MetaController)
    evolver: RuleEvolver = field(default_factory=RuleEvolver)
    neuromod: NeuromodulatorState = field(default_factory=NeuromodulatorState)
    step_count: int = 0
    rule_changes: int = 0
    evolution_events: int = 0
    performance_log: List[Dict[str, float]] = field(default_factory=list)

    def step(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Process one timestep of meta-plasticity.

        1. Observe metrics
        2. Every meta_interval: run meta-controller
        3. Every evolve_interval: run evolutionary step
        4. Apply neuromodulatory modulation
        """
        self.step_count += 1
        result: Dict[str, Any] = {"step": self.step_count, "signals": [], "evolved": False}

        # 1. Observe
        self.controller.observe(metrics)

        # 2. Meta-control
        if self.step_count % self.config.meta_interval == 0:
            signals = self.controller.decide()
            self.controller.apply_signals(self.rules, signals)
            self.rule_changes += sum(1 for s in signals if s.signal_type != MetaSignalType.NO_OP)
            result["signals"] = [s.signal_type.value for s in signals]

        # 3. Neuromodulation
        if self.config.enable_neuromodulation:
            self.neuromod.update(
                metrics.get("novelty", 0.5),
                metrics.get("surprise", 0.0),
                metrics.get("gci", 0.5),
            )
            lr_mod = self.neuromod.modulation_factor("lr")
            self.rules.stdp.lr = min(0.1, self.rules.stdp.lr * lr_mod)

        # 4. Evolution
        if self.config.enable_evolution and self.step_count % self.config.evolve_interval == 0:
            for candidate in self.evolver.population:
                self.evolver.evaluate_fitness(candidate, metrics)
            self.evolver.evolve()
            best = self.evolver.best
            if best.fitness > self.rules.fitness:
                self.rules = best.copy()
                self.evolution_events += 1
            result["evolved"] = True

        # 5. Log
        entry = {
            "step": self.step_count,
            "stdp_lr": self.rules.stdp.lr,
            "stdp_tau_plus": self.rules.stdp.tau_plus,
            "homeostatic_gain": self.rules.homeostatic.current_gain,
            "fitness": self.rules.fitness,
        }
        self.performance_log.append(entry)
        result["current_rules"] = entry

        return result

    def status(self) -> Dict[str, Any]:
        return {
            "step": self.step_count,
            "rule_changes": self.rule_changes,
            "evolution_events": self.evolution_events,
            "evolver_generation": self.evolver.generation,
            "evolver_mean_fitness": self.evolver.mean_fitness,
            "best_fitness": self.evolver.best.fitness,
            "current_stdp_lr": self.rules.stdp.lr,
            "current_tau_plus": self.rules.stdp.tau_plus,
            "neuromod_dopamine": self.neuromod.levels[NeuromodulatorType.DOPAMINE],
            "neuromod_serotonin": self.neuromod.levels[NeuromodulatorType.SEROTONIN],
        }


# ── Rule Checkpoint (Gap 1) ──────────────────────────────────────────


@dataclass
class RuleCheckpoint:
    """Serialisable snapshot of a PlasticityRuleSet at a point in time."""

    step: int
    vector: np.ndarray[Any, Any]
    fitness: float
    generation: int
    tag: str = ""

    def restore(self) -> PlasticityRuleSet:
        rs = PlasticityRuleSet.from_vector(self.vector, gen=self.generation)
        rs.fitness = self.fitness
        return rs


@dataclass
class CheckpointStore:
    """Persistent store for rule checkpoints."""

    checkpoints: List[RuleCheckpoint] = field(default_factory=list)
    max_checkpoints: int = 50

    def save(self, rules: PlasticityRuleSet, step: int, tag: str = "") -> RuleCheckpoint:
        cp = RuleCheckpoint(
            step=step,
            vector=rules.to_vector().copy(),
            fitness=rules.fitness,
            generation=rules.generation,
            tag=tag,
        )
        self.checkpoints.append(cp)
        if len(self.checkpoints) > self.max_checkpoints:
            self.checkpoints.pop(0)
        return cp

    def restore_best(self) -> Optional[PlasticityRuleSet]:
        if not self.checkpoints:
            return None
        best = max(self.checkpoints, key=lambda c: c.fitness)
        return best.restore()

    def restore_by_tag(self, tag: str) -> Optional[PlasticityRuleSet]:
        for cp in reversed(self.checkpoints):
            if cp.tag == tag:
                return cp.restore()
        return None

    @property
    def count(self) -> int:
        return len(self.checkpoints)


# ── Catastrophic Forgetting Protection (Gap 2) ──────────────────────


@dataclass
class EWCProtection:
    """Elastic Weight Consolidation for plasticity parameters.

    Penalises large deviations from previously learned rule vectors
    to protect against catastrophic forgetting.
    """

    importance: float = 1000.0
    anchor: Optional[np.ndarray[Any, Any]] = None
    fisher: Optional[np.ndarray[Any, Any]] = None

    def consolidate(self, rules: PlasticityRuleSet) -> None:
        """Set the current rules as the anchor point."""
        self.anchor = rules.to_vector().copy()
        self.fisher = np.ones_like(self.anchor)

    def penalty(self, rules: PlasticityRuleSet) -> float:
        """Compute EWC penalty for deviating from anchor."""
        if self.anchor is None or self.fisher is None:
            return 0.0
        diff = rules.to_vector() - self.anchor
        return float(0.5 * self.importance * np.sum(self.fisher * diff**2))

    def regularise(self, rules: PlasticityRuleSet, max_penalty: float = 10.0) -> PlasticityRuleSet:
        """Pull rules back toward anchor if penalty exceeds threshold."""
        if self.anchor is None:
            return rules
        pen = self.penalty(rules)
        if pen > max_penalty:
            blend = max_penalty / pen
            v = rules.to_vector()
            v_new = v * blend + self.anchor * (1.0 - blend)
            return PlasticityRuleSet.from_vector(v_new, gen=rules.generation)
        return rules


# ── Curiosity-Driven Exploration (Gap 3) ─────────────────────────────


@dataclass
class CuriositySignal:
    """Intrinsic curiosity based on prediction error of network state.

    Tracks expected next-state and computes curiosity as the prediction
    error magnitude. High curiosity → explore (increase meta-plasticity).
    """

    alpha: float = 0.1  # EMA smoothing
    _predicted: Optional[np.ndarray[Any, Any]] = None
    curiosity: float = 0.0

    def update(self, state_vector: np.ndarray[Any, Any]) -> float:
        """Update prediction model and return curiosity score."""
        if self._predicted is None:
            self._predicted = state_vector.copy()
            self.curiosity = 1.0
            return self.curiosity

        error = float(np.mean((state_vector - self._predicted) ** 2))
        self.curiosity = min(error, 1.0)
        self._predicted = self.alpha * state_vector + (1 - self.alpha) * self._predicted
        return self.curiosity


# ── Meta-Learning Rate (Gap 4) ───────────────────────────────────────


@dataclass
class MetaLearningRate:
    """Learning rate of the learning rate.

    Adapts the meta-controller sensitivity based on whether recent
    rule changes improved fitness.
    """

    meta_lr: float = 0.01
    min_meta_lr: float = 1e-4
    max_meta_lr: float = 0.5
    improvement_history: List[float] = field(default_factory=list)

    def update(self, fitness_delta: float) -> float:
        """Adjust meta_lr from fitness delta. Positive = good → speed up."""
        self.improvement_history.append(fitness_delta)
        if fitness_delta > 0:
            self.meta_lr *= 1.1
        else:
            self.meta_lr *= 0.9
        self.meta_lr = max(self.min_meta_lr, min(self.max_meta_lr, self.meta_lr))
        return self.meta_lr


# ── Sleep / Consolidation Phase (Gap 5) ──────────────────────────────


@dataclass
class SleepPhase:
    """Offline memory consolidation via experience replay.

    Stores recent metric snapshots and replays them during sleep to
    stabilise rule sets without new input.
    """

    replay_buffer: Deque[Dict[str, float]] = field(default_factory=lambda: deque(maxlen=200))
    consolidation_rounds: int = 10
    is_sleeping: bool = False

    def record(self, metrics: Dict[str, float]) -> None:
        self.replay_buffer.append(metrics)

    def sleep(self, engine_step_fn: Callable[[Dict[str, float]], Any]) -> int:
        """Run consolidation by replaying buffered experiences.

        engine_step_fn: callable that takes metrics dict.
        Returns number of replays executed.
        """
        self.is_sleeping = True
        replays = 0
        buffer_list = list(self.replay_buffer)
        for i in range(min(self.consolidation_rounds, len(buffer_list))):
            engine_step_fn(buffer_list[i])
            replays += 1
        self.is_sleeping = False
        return replays

    @property
    def buffer_size(self) -> int:
        return len(self.replay_buffer)


# ── Synaptic Tagging & Capture (Gap 6) ───────────────────────────────


@dataclass
class SynapticTag:
    """Synaptic tag for early→late LTP conversion."""

    synapse_id: int
    tag_strength: float  # 0–1
    tag_time_ms: float
    captured: bool = False

    @property
    def is_expired(self) -> bool:
        return self.tag_strength < 0.01


@dataclass
class TaggingModel:
    """Synaptic tagging and capture model.

    Early-phase LTP creates a tag. If a consolidation signal arrives
    before the tag decays, it is "captured" into late-phase LTP.
    """

    tag_decay_rate: float = 0.01
    capture_threshold: float = 0.3
    tags: List[SynapticTag] = field(default_factory=list)

    def create_tag(self, synapse_id: int, strength: float, time_ms: float) -> SynapticTag:
        tag = SynapticTag(synapse_id=synapse_id, tag_strength=strength, tag_time_ms=time_ms)
        self.tags.append(tag)
        return tag

    def decay_tags(self, dt_ms: float) -> None:
        for tag in self.tags:
            if not tag.captured:
                tag.tag_strength *= math.exp(-self.tag_decay_rate * dt_ms)

    def consolidate(self, consolidation_strength: float) -> int:
        """Attempt capture on all active tags. Returns count of captured."""
        captured = 0
        for tag in self.tags:
            if not tag.captured and tag.tag_strength >= self.capture_threshold:
                if consolidation_strength > 0.5:
                    tag.captured = True
                    captured += 1
        return captured

    def prune_expired(self) -> int:
        before = len(self.tags)
        self.tags = [t for t in self.tags if not t.is_expired or t.captured]
        return before - len(self.tags)

    @property
    def active_tags(self) -> int:
        return sum(1 for t in self.tags if not t.captured and not t.is_expired)


# ── Population Diversity Tracker (Gap 7) ─────────────────────────────


def population_diversity(evolver: RuleEvolver) -> float:
    """Compute diversity as mean pairwise L2 distance of rule vectors."""
    vectors = [r.to_vector() for r in evolver.population]
    n = len(vectors)
    if n < 2:
        return 0.0
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += float(np.linalg.norm(vectors[i] - vectors[j]))
            count += 1
    return total / count


def inject_diversity(evolver: RuleEvolver, n_random: int = 2) -> None:
    """Replace worst individuals with fresh random rule sets."""
    sorted_pop = sorted(evolver.population, key=lambda r: r.fitness)
    for i in range(min(n_random, len(sorted_pop))):
        v = evolver.rng.normal(0, 1, size=sorted_pop[i].vector_dim)
        v[:5] = np.abs(v[:5]) * 10 + 1  # keep STDP params positive
        sorted_pop[i] = PlasticityRuleSet.from_vector(
            PlasticityRuleSet().to_vector() + v * 0.1,
            gen=evolver.generation,
        )
    evolver.population = sorted_pop


# ── Context-Dependent Rule Switching (Gap 8) ─────────────────────────


@dataclass
class ContextRuleBank:
    """Maintains separate rule sets per context/task.

    Allows rapid switching between learned plasticity configurations
    without catastrophic interference.
    """

    bank: Dict[str, PlasticityRuleSet] = field(default_factory=dict)
    active_context: str = "default"

    def store(self, context: str, rules: PlasticityRuleSet) -> None:
        self.bank[context] = rules.copy()

    def switch(self, context: str) -> Optional[PlasticityRuleSet]:
        self.active_context = context
        if context in self.bank:
            return self.bank[context].copy()
        return None

    def contexts(self) -> List[str]:
        return list(self.bank.keys())

    @property
    def num_contexts(self) -> int:
        return len(self.bank)


# ── Fitness Trajectory Analysis (Gap 9) ──────────────────────────────


@dataclass
class FitnessTrajectory:
    """Tracks fitness over time and detects trends."""

    history: List[float] = field(default_factory=list)
    window: int = 20

    def record(self, fitness: float) -> None:
        self.history.append(fitness)

    def trend(self) -> float:
        """Return slope of fitness over recent window. >0 = improving."""
        if len(self.history) < 2:
            return 0.0
        recent = self.history[-self.window :]
        x = np.arange(len(recent), dtype=float)
        y = np.array(recent)
        if np.std(x) == 0:
            return 0.0
        slope = float(np.polyfit(x, y, 1)[0])
        return slope

    @property
    def is_improving(self) -> bool:
        return self.trend() > 0

    @property
    def is_stagnant(self) -> bool:
        if len(self.history) < self.window:
            return False
        recent = self.history[-self.window :]
        return float(np.std(recent)) < 1e-4

    @property
    def best_ever(self) -> float:
        return max(self.history) if self.history else 0.0


# ── Rule Constraint Enforcement (Gap 10) ─────────────────────────────


@dataclass
class RuleConstraints:
    """Hard constraints on plasticity parameters to prevent pathological values."""

    stdp_lr_range: Tuple[float, float] = (1e-6, 0.1)
    stdp_tau_range: Tuple[float, float] = (1.0, 100.0)
    stp_u_range: Tuple[float, float] = (0.01, 0.99)
    homeostatic_target_range: Tuple[float, float] = (0.1, 100.0)
    bitstream_length_range: Tuple[int, int] = (32, 4096)

    def enforce(self, rules: PlasticityRuleSet) -> PlasticityRuleSet:
        """Clamp all parameters to valid ranges."""
        rules.stdp.lr = max(self.stdp_lr_range[0], min(self.stdp_lr_range[1], rules.stdp.lr))
        rules.stdp.tau_plus = max(
            self.stdp_tau_range[0], min(self.stdp_tau_range[1], rules.stdp.tau_plus)
        )
        rules.stdp.tau_minus = max(
            self.stdp_tau_range[0], min(self.stdp_tau_range[1], rules.stdp.tau_minus)
        )
        rules.stdp.a_plus = max(1e-6, rules.stdp.a_plus)
        rules.stdp.a_minus = max(1e-6, rules.stdp.a_minus)
        rules.stp.u_base = max(self.stp_u_range[0], min(self.stp_u_range[1], rules.stp.u_base))
        rules.homeostatic.target_rate_hz = max(
            self.homeostatic_target_range[0],
            min(self.homeostatic_target_range[1], rules.homeostatic.target_rate_hz),
        )
        rules.bitstream.length = max(
            self.bitstream_length_range[0],
            min(self.bitstream_length_range[1], rules.bitstream.length),
        )
        return rules

    def is_valid(self, rules: PlasticityRuleSet) -> bool:
        """Check if all parameters are within constraints."""
        lr = rules.stdp.lr
        if not (self.stdp_lr_range[0] <= lr <= self.stdp_lr_range[1]):
            return False
        tau = rules.stdp.tau_plus
        if not (self.stdp_tau_range[0] <= tau <= self.stdp_tau_range[1]):
            return False
        u = rules.stp.u_base
        return self.stp_u_range[0] <= u <= self.stp_u_range[1]
