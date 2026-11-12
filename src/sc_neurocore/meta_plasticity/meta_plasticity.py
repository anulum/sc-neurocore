# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Self-Evolving Meta-Plasticity SC Engine

"""Plasticity parameters that a rule-based controller and an evolver change at runtime.

The module holds STDP, short-term plasticity, homeostatic and bitstream
parameters as one mutable rule set, and changes them from performance
metrics that the caller passes in as dictionaries (the keys ``novelty``,
``surprise`` and ``gci``, and for fitness ``gci_std``, ``mean_surprise`` and
``mean_rate_hz``). The controller compares running means and spreads of
those metrics with fixed thresholds; the evolver selects, recombines and
mutates parameter vectors. Around them are checkpoints, a quadratic anchor
penalty, a curiosity score, replay, synaptic tagging, diversity helpers,
per-context rule banks, fitness trends and parameter limits.

The module computes on floating-point numbers; it does not encode the
controller's inputs or decisions as bitstreams, and it imports no neuron,
layer or synapse model: applying a rule set to a network is the caller's.

Architecture:

    metrics dict ──► MetaController ──► signals ──► PlasticityRuleSet
                       (thresholds)                    ├── STDP (tau, A, lr)
    metrics dict ──► RuleEvolver (fitness, selection,  ├── STP  (U, tau_d/f)
                       crossover, mutation) ─────────► ├── Homeostatic
                                                       └── Bitstream (length)

- **PlasticityRuleSet**: Parametric STDP/STP/homeostatic rules as mutable
  dataclass, with an eleven-number vector form for checkpoints and
  evolution.
- **MetaController**: observes metrics and emits rule modifications.
- **RuleEvolver**: Evolutionary engine that maintains a population of
  candidate rule sets, selects based on fitness, and recombines.
- **MetaPlasticityEngine**: steps the controller, a neuromodulatory scaling
  of the learning rate and the evolver on a schedule.
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
    """Mutable STDP parameters.

    Attributes
    ----------
    tau_plus, tau_minus : float
        Potentiation and depression time constants in milliseconds.
    a_plus, a_minus : float
        Potentiation and depression amplitudes.
    lr : float
        Learning rate.
    """

    tau_plus: float = 20.0  # ms
    tau_minus: float = 20.0  # ms
    a_plus: float = 0.01
    a_minus: float = 0.012
    lr: float = 0.01

    def to_vector(self) -> np.ndarray[Any, Any]:
        """Return the five parameters as ``[tau_plus, tau_minus, a_plus, a_minus, lr]``."""
        return np.array([self.tau_plus, self.tau_minus, self.a_plus, self.a_minus, self.lr])

    @classmethod
    def from_vector(cls, v: np.ndarray[Any, Any]) -> STDPParams:
        """Build parameters from the first five numbers of a vector.

        Parameters
        ----------
        v : numpy.ndarray
            ``[tau_plus, tau_minus, a_plus, a_minus, lr]`` and possibly more.

        Returns
        -------
        STDPParams
            Time constants floored at 1.0 and the other three at 1e-6; no
            upper limit is applied.
        """
        return cls(
            tau_plus=max(1.0, float(v[0])),
            tau_minus=max(1.0, float(v[1])),
            a_plus=max(1e-6, float(v[2])),
            a_minus=max(1e-6, float(v[3])),
            lr=max(1e-6, float(v[4])),
        )


@dataclass
class STPParams:
    """Mutable short-term plasticity parameters.

    Attributes
    ----------
    u_base : float
        Baseline release probability.
    tau_d : float
        Depression time constant in milliseconds.
    tau_f : float
        Facilitation time constant in milliseconds.
    """

    u_base: float = 0.5
    tau_d: float = 200.0  # ms, depression
    tau_f: float = 20.0  # ms, facilitation

    def to_vector(self) -> np.ndarray[Any, Any]:
        """Return the three parameters as ``[u_base, tau_d, tau_f]``."""
        return np.array([self.u_base, self.tau_d, self.tau_f])

    @classmethod
    def from_vector(cls, v: np.ndarray[Any, Any]) -> STPParams:
        """Build parameters from the first three numbers of a vector.

        Parameters
        ----------
        v : numpy.ndarray
            ``[u_base, tau_d, tau_f]`` and possibly more.

        Returns
        -------
        STPParams
            ``u_base`` clipped to [0.01, 0.99] and both time constants floored
            at 1.0.
        """
        return cls(
            u_base=float(np.clip(v[0], 0.01, 0.99)),
            tau_d=max(1.0, float(v[1])),
            tau_f=max(1.0, float(v[2])),
        )


@dataclass
class HomeostaticParams:
    """Homeostatic plasticity: target rate + gain modulation.

    Attributes
    ----------
    target_rate_hz : float
        Firing rate the gain pushes towards.
    gain_adaptation_rate : float
        Gain change per hertz of rate error.
    current_gain : float
        The present gain, kept in [0.1, 10] by :meth:`adapt`; the
        constructor accepts any value.
    """

    target_rate_hz: float = 5.0
    gain_adaptation_rate: float = 0.001
    current_gain: float = 1.0

    def adapt(self, measured_rate_hz: float) -> float:
        """Adjust gain to push firing rate toward target.

        Parameters
        ----------
        measured_rate_hz : float
            The observed firing rate.

        Returns
        -------
        float
            The new gain: the old one plus rate × (target - measured), clamped
            to [0.1, 10].
        """
        error = self.target_rate_hz - measured_rate_hz
        self.current_gain += self.gain_adaptation_rate * error
        self.current_gain = max(0.1, min(10.0, self.current_gain))
        return self.current_gain


@dataclass
class BitstreamParams:
    """Mutable SC bitstream parameters.

    Attributes
    ----------
    length : int
        Bitstream length; the only one of the three in the vector form.
    lfsr_seed : int
        LFSR seed.
    precision_bits : int
        Fractional bits of the fixed-point format.
    """

    length: int = 256
    lfsr_seed: int = 0xACE1
    precision_bits: int = 8  # Q8.8


@dataclass
class PlasticityRuleSet:
    """Complete set of mutable plasticity rules.

    This is the "genome" that the meta-controller evolves.

    Attributes
    ----------
    stdp : STDPParams
        STDP parameters.
    stp : STPParams
        Short-term plasticity parameters.
    homeostatic : HomeostaticParams
        Homeostatic parameters.
    bitstream : BitstreamParams
        Bitstream parameters.
    generation : int
        Generation of the evolver that made the set.
    fitness : float
        The last fitness assigned to the set.
    """

    stdp: STDPParams = field(default_factory=STDPParams)
    stp: STPParams = field(default_factory=STPParams)
    homeostatic: HomeostaticParams = field(default_factory=HomeostaticParams)
    bitstream: BitstreamParams = field(default_factory=BitstreamParams)
    generation: int = 0
    fitness: float = 0.0

    def to_vector(self) -> np.ndarray[Any, Any]:
        """Serialise all mutable params to a flat vector.

        Eleven numbers: the five STDP and three STP parameters, the target
        rate, the gain adaptation rate and the bitstream length. The current
        gain, the LFSR seed, the precision, the generation and the fitness
        are not part of it.
        """
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
        """Build a rule set from its eleven-number vector.

        Parameters
        ----------
        v : numpy.ndarray
            A vector in the order of :meth:`to_vector`.
        gen : int
            Generation of the new set.

        Returns
        -------
        PlasticityRuleSet
            The set with the floors of the parameter classes, a target rate of
            at least 0.1, an adaptation rate of at least 1e-6 and a length of
            at least 32 (truncated to an integer). The values outside the
            vector start from their defaults: gain 1.0, the default seed and
            precision, fitness 0.0.
        """
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
        """Length of the vector form, 11."""
        return len(self.to_vector())

    def copy(self) -> PlasticityRuleSet:
        """Return a deep copy, fitness and generation included."""
        return copy.deepcopy(self)


# ── Meta-Controller ──────────────────────────────────────────────────


class MetaSignalType(Enum):
    """Types of meta-control signals.

    :meth:`MetaController.decide` emits ``INCREASE_LR``, ``DECREASE_LR``,
    ``WIDEN_WINDOW``, ``NARROW_WINDOW`` and ``NO_OP``;
    ``INCREASE_HOMEOSTATIC`` and ``RESET_STP`` are applied by
    :meth:`MetaController.apply_signals` but only reach it from a caller.
    """

    INCREASE_LR = "increase_lr"
    DECREASE_LR = "decrease_lr"
    WIDEN_WINDOW = "widen_window"
    NARROW_WINDOW = "narrow_window"
    INCREASE_HOMEOSTATIC = "increase_homeostatic"
    RESET_STP = "reset_stp"
    NO_OP = "no_op"


@dataclass
class MetaControlSignal:
    """One meta-control directive.

    Attributes
    ----------
    signal_type : MetaSignalType
        What to change.
    magnitude : float
        Size of the change; its meaning depends on the type.
    target_param : str
        Name of the changed parameter, informational only.
    """

    signal_type: MetaSignalType
    magnitude: float = 0.1
    target_param: str = ""


class MetaController:
    """SC-domain meta-controller that rewrites plasticity rules.

    Observes network performance metrics (surprise, novelty, GCI) and emits
    rule modifications. Its decisions compare means and a standard deviation
    of the recent metrics with fixed thresholds on floating-point numbers;
    no bitstream is involved.

    Parameters
    ----------
    sensitivity : float
        Scale of the novelty and GCI-spread thresholds.
    rng_seed : int
        Seed of ``rng``.

    Attributes
    ----------
    sensitivity : float
        Scale of the thresholds.
    rng : numpy.random.Generator
        A generator the controller does not use.
    observation_window : collections.deque
        The last 100 observed metric dictionaries.
    signal_history : list of MetaControlSignal
        Every signal :meth:`decide` returned, in order.
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

        With fewer than five observations it returns one ``NO_OP``. Otherwise,
        over the last ten observations (missing keys read as novelty 0.5,
        surprise 0.0, GCI 0.5):

        - mean novelty above 0.7 × sensitivity → increase the learning rate
          by the factor 1 + 0.1 × mean novelty;
        - else mean novelty below 0.3 and mean surprise below 0.1 → decrease
          it by the factor 1 - 0.05 (consolidate);
        - GCI standard deviation above 0.1 × sensitivity → widen the STDP
          window by 2 ms (explore);
        - else a standard deviation below 0.02 with mean GCI above 0.7 →
          narrow it by 1 ms (exploit).

        Firing rates are not read. ``NO_OP`` stands in when no rule applies.

        Returns
        -------
        list of MetaControlSignal
            The signals, also appended to ``signal_history`` (the early
            ``NO_OP`` is not).
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
        """Apply meta-control signals to a rule set (in-place).

        Learning-rate signals scale ``lr`` by 1 ± magnitude, capped at 0.1 or
        floored at 1e-6; window signals move both STDP time constants by the
        magnitude, capped at 100 or floored at 5 ms; ``INCREASE_HOMEOSTATIC``
        multiplies the gain adaptation rate by 1.5; ``RESET_STP`` restores the
        default short-term parameters.

        Parameters
        ----------
        rules : PlasticityRuleSet
            The rule set to change.
        signals : list of MetaControlSignal
            The signals, applied in order.

        Returns
        -------
        PlasticityRuleSet
            The same object, changed.
        """
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

    Attributes
    ----------
    population_size : int
        Size of each generation; parent selection needs at least four.
    mutation_rate : float
        Probability that a vector entry mutates.
    mutation_scale : float
        Standard deviation of a mutation relative to the entry's size.
    elite_count : int
        Best sets copied unchanged into the next generation.
    rng : numpy.random.Generator
        Source of selection, crossover and mutation draws.
    population : list of PlasticityRuleSet
        The current generation; empty means ``population_size`` default
        sets.
    generation : int
        Generations evolved so far.
    """

    population_size: int = 8
    mutation_rate: float = 0.1
    mutation_scale: float = 0.05
    elite_count: int = 2
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(42))
    population: List[PlasticityRuleSet] = field(default_factory=list)
    generation: int = 0

    def __post_init__(self) -> None:
        """Fill an empty population with default rule sets."""
        if not self.population:
            self.population = [PlasticityRuleSet(generation=0) for _ in range(self.population_size)]

    def evaluate_fitness(self, rules: PlasticityRuleSet, metrics: Dict[str, float]) -> float:
        """Compute fitness from network performance metrics.

        Fitness = GCI × max(stability, 0) - 0.3 × surprise - 0.2 × rate
        penalty, with stability 1 - ``gci_std`` and the rate penalty
        |``mean_rate_hz`` - target rate| / 10 capped at 1. Missing keys read as
        GCI 0.5, ``gci_std`` 0.1, surprise 0.0 and rate 5 Hz. Only the rule
        set's target rate enters, so under one set of metrics the candidates
        differ in fitness by their target rates alone.

        Parameters
        ----------
        rules : PlasticityRuleSet
            The candidate; its ``fitness`` is set.
        metrics : dict of str to float
            ``gci``, ``gci_std``, ``mean_surprise`` and ``mean_rate_hz``.

        Returns
        -------
        float
            The fitness.
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
        """Tournament selection of two parents.

        Four distinct members are drawn and the two fittest returned.

        Returns
        -------
        tuple of PlasticityRuleSet
            The fittest and the second fittest of the four.

        Raises
        ------
        ValueError
            If the population has fewer than four members.
        """
        candidates = self.rng.choice(len(self.population), size=4, replace=False)
        sorted_c = sorted(candidates, key=lambda i: self.population[i].fitness, reverse=True)
        return self.population[sorted_c[0]], self.population[sorted_c[1]]

    def crossover(self, p1: PlasticityRuleSet, p2: PlasticityRuleSet) -> PlasticityRuleSet:
        """Uniform crossover of two rule sets.

        Each of the eleven vector entries comes from either parent with
        probability 0.5.

        Parameters
        ----------
        p1, p2 : PlasticityRuleSet
            The parents.

        Returns
        -------
        PlasticityRuleSet
            The child, of the next generation, with fitness 0.0.
        """
        v1 = p1.to_vector()
        v2 = p2.to_vector()
        mask = self.rng.random(len(v1)) < 0.5
        child_v = np.where(mask, v1, v2)
        return PlasticityRuleSet.from_vector(child_v, gen=self.generation + 1)

    def mutate(self, rules: PlasticityRuleSet) -> PlasticityRuleSet:
        """Gaussian mutation of rule parameters.

        Each vector entry mutates with probability ``mutation_rate`` by
        N(0, ``mutation_scale``) times its own size, so a change is relative.

        Parameters
        ----------
        rules : PlasticityRuleSet
            The set to mutate; it is not changed.

        Returns
        -------
        PlasticityRuleSet
            A new set of the next generation, with fitness 0.0.
        """
        v = rules.to_vector()
        mask = self.rng.random(len(v)) < self.mutation_rate
        noise = self.rng.normal(0, self.mutation_scale, size=len(v))
        v[mask] += noise[mask] * np.abs(v[mask] + 1e-8)
        return PlasticityRuleSet.from_vector(v, gen=self.generation + 1)

    def evolve(self) -> List[PlasticityRuleSet]:
        """Run one generation of evolution.

        Copies of the ``elite_count`` fittest sets are kept; the rest are
        mutated crossover children of tournament-selected parents.

        Returns
        -------
        list of PlasticityRuleSet
            The new population, also stored.
        """
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
        """The population's member with the highest stored fitness."""
        return max(self.population, key=lambda r: r.fitness)

    @property
    def mean_fitness(self) -> float:
        """Mean stored fitness of the population; new children count 0.0."""
        return float(np.mean([r.fitness for r in self.population]))


# ── Neuromodulatory State ────────────────────────────────────────────


class NeuromodulatorType(Enum):
    """The four modelled neuromodulators.

    Each is a level in [0, 1] of :class:`NeuromodulatorState`, not a
    concentration.
    """

    DOPAMINE = "dopamine"  # reward / surprise
    SEROTONIN = "serotonin"  # mood / stability
    ACETYLCHOLINE = "acetylcholine"  # attention / precision
    NOREPINEPHRINE = "norepinephrine"  # arousal / exploration


@dataclass
class NeuromodulatorState:
    """Simulated neuromodulatory tone that modulates meta-plasticity.

    Attributes
    ----------
    levels : dict of NeuromodulatorType to float
        Each modulator's level in [0, 1], 0.5 at the start.
    decay_rate : float
        Amount subtracted from every level at each update.
    """

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
        """Update neuromodulator levels from network state.

        Dopamine moves by 0.1 × (surprise - 0.5), serotonin by 0.05 × (GCI -
        0.5), acetylcholine by 0.08 × (novelty - 0.5) and norepinephrine by
        0.06 × (surprise - 0.3); each also loses ``decay_rate`` and is clamped
        to [0, 1].

        Parameters
        ----------
        novelty, surprise, gci : float
            The current metrics.
        """
        self.levels[NeuromodulatorType.DOPAMINE] += 0.1 * (surprise - 0.5) - self.decay_rate
        self.levels[NeuromodulatorType.SEROTONIN] += 0.05 * (gci - 0.5) - self.decay_rate
        self.levels[NeuromodulatorType.ACETYLCHOLINE] += 0.08 * (novelty - 0.5) - self.decay_rate
        self.levels[NeuromodulatorType.NOREPINEPHRINE] += 0.06 * (surprise - 0.3) - self.decay_rate

        for nm in self.levels:
            self.levels[nm] = max(0.0, min(1.0, self.levels[nm]))

    def modulation_factor(self, param: str) -> float:
        """Get a combined modulation factor for a specific parameter type.

        Parameters
        ----------
        param : str
            ``"lr"``, ``"tau"`` or ``"gain"``.

        Returns
        -------
        float
            0.5 + dopamine + 0.3 × norepinephrine for ``"lr"`` (0.5 to 1.8),
            0.8 + 0.4 × (1 - acetylcholine) for ``"tau"``, 0.5 + 0.5 ×
            norepinephrine for ``"gain"``, and 1.0 otherwise. Serotonin enters
            none of them.
        """
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
    """Configuration for the meta-plasticity engine.

    Attributes
    ----------
    num_neurons : int
        Stored and not used.
    meta_interval : int
        Steps between controller decisions.
    evolve_interval : int
        Steps between evolution rounds.
    enable_neuromodulation : bool
        Whether each step scales the learning rate by the neuromodulators.
    enable_evolution : bool
        Whether evolution rounds run.
    """

    num_neurons: int = 16
    meta_interval: int = 50
    evolve_interval: int = 500
    enable_neuromodulation: bool = True
    enable_evolution: bool = True


@dataclass
class MetaPlasticityEngine:
    """Top-level orchestrator for self-evolving meta-plasticity.

    Steps the meta-controller, the neuromodulatory scaling of the learning
    rate and the rule evolver on one rule set; it holds no neurons.

    Attributes
    ----------
    config : EngineConfig
        Schedules and switches.
    rules : PlasticityRuleSet
        The active rule set.
    controller : MetaController
        The rule-based controller.
    evolver : RuleEvolver
        The evolver.
    neuromod : NeuromodulatorState
        The neuromodulator levels.
    step_count : int
        Steps taken.
    rule_changes : int
        Signals other than ``NO_OP`` applied.
    evolution_events : int
        Times the evolver's best replaced the active rule set.
    performance_log : list of dict
        One entry per step, never trimmed.
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
        3. With neuromodulation on: update the levels and multiply the
           learning rate by the ``"lr"`` factor (0.5 to 1.8), capped at 0.1
           with no lower limit, so low tone shrinks it step after step
        4. Every evolve_interval (with evolution on): give every candidate
           this step's fitness, evolve one generation, and adopt a copy of
           the best when its fitness exceeds the active set's

        Parameters
        ----------
        metrics : dict of str to float
            The step's metrics.

        Returns
        -------
        dict of str to Any
            The step number, the emitted signal names, whether evolution ran,
            and the logged entry (learning rate, ``tau_plus``, gain, fitness).
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
        """Return the engine's counters, the evolver's state and the current rates.

        Returns
        -------
        dict of str to Any
            Steps, rule changes, evolution events, the evolver's generation,
            mean and best fitness, the learning rate, ``tau_plus`` and the
            dopamine and serotonin levels.
        """
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
    """Serialisable snapshot of a PlasticityRuleSet at a point in time.

    Attributes
    ----------
    step : int
        Step at which it was taken.
    vector : numpy.ndarray
        The rule set's eleven-number vector.
    fitness : float
        The set's fitness.
    generation : int
        The set's generation.
    tag : str
        A label.
    """

    step: int
    vector: np.ndarray[Any, Any]
    fitness: float
    generation: int
    tag: str = ""

    def restore(self) -> PlasticityRuleSet:
        """Rebuild the rule set from the vector, with the stored fitness and generation.

        Values outside the vector (current gain, seed, precision) come back
        as defaults.
        """
        rs = PlasticityRuleSet.from_vector(self.vector, gen=self.generation)
        rs.fitness = self.fitness
        return rs


@dataclass
class CheckpointStore:
    """In-memory store for rule checkpoints.

    Nothing is written to disk; when full, the oldest checkpoint is dropped.

    Attributes
    ----------
    checkpoints : list of RuleCheckpoint
        The checkpoints, oldest first.
    max_checkpoints : int
        Largest number kept.
    """

    checkpoints: List[RuleCheckpoint] = field(default_factory=list)
    max_checkpoints: int = 50

    def save(self, rules: PlasticityRuleSet, step: int, tag: str = "") -> RuleCheckpoint:
        """Store a checkpoint of a rule set.

        Parameters
        ----------
        rules : PlasticityRuleSet
            The set to snapshot.
        step : int
            Current step.
        tag : str
            A label.

        Returns
        -------
        RuleCheckpoint
            The stored checkpoint.
        """
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
        """Restore the checkpoint of highest fitness; ``None`` when empty."""
        if not self.checkpoints:
            return None
        best = max(self.checkpoints, key=lambda c: c.fitness)
        return best.restore()

    def restore_by_tag(self, tag: str) -> Optional[PlasticityRuleSet]:
        """Restore the newest checkpoint with a tag; ``None`` when there is none."""
        for cp in reversed(self.checkpoints):
            if cp.tag == tag:
                return cp.restore()
        return None

    @property
    def count(self) -> int:
        """Number of stored checkpoints."""
        return len(self.checkpoints)


# ── Catastrophic Forgetting Protection (Gap 2) ──────────────────────


@dataclass
class EWCProtection:
    """Elastic Weight Consolidation for plasticity parameters.

    Penalises large deviations from previously learned rule vectors
    to protect against catastrophic forgetting. The importance weights are
    all 1, not a Fisher information estimate, so the penalty is a plain
    squared distance over the raw vector, in which the bitstream length and
    the time constants dominate the small rates.

    Attributes
    ----------
    importance : float
        Scale of the penalty.
    anchor : numpy.ndarray or None
        The consolidated vector.
    fisher : numpy.ndarray or None
        Per-entry weights; ones after :meth:`consolidate`.
    """

    importance: float = 1000.0
    anchor: Optional[np.ndarray[Any, Any]] = None
    fisher: Optional[np.ndarray[Any, Any]] = None

    def consolidate(self, rules: PlasticityRuleSet) -> None:
        """Set the current rules as the anchor point."""
        self.anchor = rules.to_vector().copy()
        self.fisher = np.ones_like(self.anchor)

    def penalty(self, rules: PlasticityRuleSet) -> float:
        """Compute EWC penalty for deviating from anchor.

        Returns
        -------
        float
            0.5 × importance × Σ weight × (v - anchor)²; 0.0 before
            :meth:`consolidate`.
        """
        if self.anchor is None or self.fisher is None:
            return 0.0
        diff = rules.to_vector() - self.anchor
        return float(0.5 * self.importance * np.sum(self.fisher * diff**2))

    def regularise(self, rules: PlasticityRuleSet, max_penalty: float = 10.0) -> PlasticityRuleSet:
        """Pull rules back toward anchor if penalty exceeds threshold.

        Parameters
        ----------
        rules : PlasticityRuleSet
            The set to check.
        max_penalty : float
            Penalty above which the set is pulled back.

        Returns
        -------
        PlasticityRuleSet
            The same object when within the limit (or before consolidation);
            otherwise a new set at blend × v + (1 - blend) × anchor with blend =
            max_penalty / penalty, whose fitness is 0.0.
        """
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
    The prediction is an exponential moving average of past states.

    Attributes
    ----------
    alpha : float
        Weight of the newest state in the moving average.
    curiosity : float
        The last score.
    """

    alpha: float = 0.1  # EMA smoothing
    _predicted: Optional[np.ndarray[Any, Any]] = None
    curiosity: float = 0.0

    def update(self, state_vector: np.ndarray[Any, Any]) -> float:
        """Update prediction model and return curiosity score.

        Parameters
        ----------
        state_vector : numpy.ndarray
            The current state.

        Returns
        -------
        float
            1.0 for the first state; afterwards the mean squared difference
            from the prediction, capped at 1.0.
        """
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

    Grows when the reported fitness change is positive and shrinks
    otherwise. Nothing in the module applies it to the meta-controller; it
    is a value for the caller.

    Attributes
    ----------
    meta_lr : float
        The current rate.
    min_meta_lr, max_meta_lr : float
        Its limits.
    improvement_history : list of float
        Every reported fitness change.
    """

    meta_lr: float = 0.01
    min_meta_lr: float = 1e-4
    max_meta_lr: float = 0.5
    improvement_history: List[float] = field(default_factory=list)

    def update(self, fitness_delta: float) -> float:
        """Adjust meta_lr from fitness delta. Positive = good → speed up.

        Parameters
        ----------
        fitness_delta : float
            The change of fitness; zero counts as no improvement.

        Returns
        -------
        float
            The rate × 1.1 after an improvement, × 0.9 otherwise, kept within
            its limits.
        """
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

    Attributes
    ----------
    replay_buffer : collections.deque
        The last 200 recorded metric dictionaries.
    consolidation_rounds : int
        Largest number of replays per sleep.
    is_sleeping : bool
        Set when :meth:`sleep` starts and cleared when it returns; a step
        that raises leaves it set.
    """

    replay_buffer: Deque[Dict[str, float]] = field(default_factory=lambda: deque(maxlen=200))
    consolidation_rounds: int = 10
    is_sleeping: bool = False

    def record(self, metrics: Dict[str, float]) -> None:
        """Add one metric snapshot to the replay buffer."""
        self.replay_buffer.append(metrics)

    def sleep(self, engine_step_fn: Callable[[Dict[str, float]], Any]) -> int:
        """Run consolidation by replaying buffered experiences.

        engine_step_fn: callable that takes metrics dict.
        Returns number of replays executed.

        The replayed snapshots are the oldest ones in the buffer, from its
        start, up to ``consolidation_rounds`` of them.
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
        """Number of snapshots in the replay buffer."""
        return len(self.replay_buffer)


# ── Synaptic Tagging & Capture (Gap 6) ───────────────────────────────


@dataclass
class SynapticTag:
    """Synaptic tag for early→late LTP conversion.

    Attributes
    ----------
    synapse_id : int
        The tagged synapse.
    tag_strength : float
        Strength from 0 to 1.
    tag_time_ms : float
        Time of tagging in milliseconds.
    captured : bool
        Whether a consolidation captured the tag.
    """

    synapse_id: int
    tag_strength: float  # 0–1
    tag_time_ms: float
    captured: bool = False

    @property
    def is_expired(self) -> bool:
        """Whether the strength has fallen below 0.01."""
        return self.tag_strength < 0.01


@dataclass
class TaggingModel:
    """Synaptic tagging and capture model.

    Early-phase LTP creates a tag. If a consolidation signal arrives
    before the tag decays, it is "captured" into late-phase LTP.

    Attributes
    ----------
    tag_decay_rate : float
        Exponential decay rate per millisecond.
    capture_threshold : float
        Least strength a tag needs to be captured.
    tags : list of SynapticTag
        All tags.
    """

    tag_decay_rate: float = 0.01
    capture_threshold: float = 0.3
    tags: List[SynapticTag] = field(default_factory=list)

    def create_tag(self, synapse_id: int, strength: float, time_ms: float) -> SynapticTag:
        """Create and keep a tag.

        Parameters
        ----------
        synapse_id : int
            The tagged synapse.
        strength : float
            Initial strength.
        time_ms : float
            Time of tagging.

        Returns
        -------
        SynapticTag
            The new tag.
        """
        tag = SynapticTag(synapse_id=synapse_id, tag_strength=strength, tag_time_ms=time_ms)
        self.tags.append(tag)
        return tag

    def decay_tags(self, dt_ms: float) -> None:
        """Multiply every uncaptured tag's strength by exp(-rate × dt_ms)."""
        for tag in self.tags:
            if not tag.captured:
                tag.tag_strength *= math.exp(-self.tag_decay_rate * dt_ms)

    def consolidate(self, consolidation_strength: float) -> int:
        """Attempt capture on all active tags. Returns count of captured.

        A consolidation strength above 0.5 captures every uncaptured tag at or
        above the capture threshold; a weaker one captures none.
        """
        captured = 0
        for tag in self.tags:
            if not tag.captured and tag.tag_strength >= self.capture_threshold:
                if consolidation_strength > 0.5:
                    tag.captured = True
                    captured += 1
        return captured

    def prune_expired(self) -> int:
        """Remove expired uncaptured tags and return how many were removed."""
        before = len(self.tags)
        self.tags = [t for t in self.tags if not t.is_expired or t.captured]
        return before - len(self.tags)

    @property
    def active_tags(self) -> int:
        """Number of tags neither captured nor expired."""
        return sum(1 for t in self.tags if not t.captured and not t.is_expired)


# ── Population Diversity Tracker (Gap 7) ─────────────────────────────


def population_diversity(evolver: RuleEvolver) -> float:
    """Compute diversity as mean pairwise L2 distance of rule vectors.

    The distance is over the raw eleven-number vectors, so the bitstream
    length and the time constants dominate it.

    Parameters
    ----------
    evolver : RuleEvolver
        The evolver whose population is measured.

    Returns
    -------
    float
        The mean distance; 0.0 for fewer than two members.
    """
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
    """Replace worst individuals with fresh random rule sets.

    Each new set is the default vector plus 0.1 × a Gaussian vector whose
    first five entries are |N(0, 1)| × 10 + 1, so its STDP time constants,
    amplitudes and learning rate grow by 0.1 or more: every new learning rate
    is at least 0.11, above the 0.1 that :class:`RuleConstraints` allows. The
    population is left in the order of the fitness before the replacement, the
    new sets (fitness 0.0) in the places of the least fit.

    Parameters
    ----------
    evolver : RuleEvolver
        The evolver to change; its generator draws the vectors.
    n_random : int
        How many of the least fit to replace.
    """
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

    Attributes
    ----------
    bank : dict of str to PlasticityRuleSet
        A copy of the rule set stored for each context.
    active_context : str
        The context last switched to.
    """

    bank: Dict[str, PlasticityRuleSet] = field(default_factory=dict)
    active_context: str = "default"

    def store(self, context: str, rules: PlasticityRuleSet) -> None:
        """Store a copy of a rule set under a context, replacing an earlier one."""
        self.bank[context] = rules.copy()

    def switch(self, context: str) -> Optional[PlasticityRuleSet]:
        """Make a context active and return a copy of its rule set.

        The context becomes active even when nothing is stored for it; the
        result is then ``None``.
        """
        self.active_context = context
        if context in self.bank:
            return self.bank[context].copy()
        return None

    def contexts(self) -> List[str]:
        """Return the stored contexts in the order first stored."""
        return list(self.bank.keys())

    @property
    def num_contexts(self) -> int:
        """Number of stored contexts."""
        return len(self.bank)


# ── Fitness Trajectory Analysis (Gap 9) ──────────────────────────────


@dataclass
class FitnessTrajectory:
    """Tracks fitness over time and detects trends.

    Attributes
    ----------
    history : list of float
        Every recorded fitness.
    window : int
        Number of recent values the trend and stagnation look at.
    """

    history: List[float] = field(default_factory=list)
    window: int = 20

    def record(self, fitness: float) -> None:
        """Append one fitness value to the history."""
        self.history.append(fitness)

    def trend(self) -> float:
        """Return slope of fitness over recent window. >0 = improving.

        The least-squares slope per record over the last ``window`` values;
        0.0 for fewer than two.
        """
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
        """Whether the trend is positive."""
        return self.trend() > 0

    @property
    def is_stagnant(self) -> bool:
        """Whether a full window's standard deviation is below 1e-4."""
        if len(self.history) < self.window:
            return False
        recent = self.history[-self.window :]
        return float(np.std(recent)) < 1e-4

    @property
    def best_ever(self) -> float:
        """The highest recorded fitness; 0.0 for an empty history."""
        return max(self.history) if self.history else 0.0


# ── Rule Constraint Enforcement (Gap 10) ─────────────────────────────


@dataclass
class RuleConstraints:
    """Hard constraints on plasticity parameters to prevent pathological values.

    Attributes
    ----------
    stdp_lr_range : tuple of float
        Limits of the STDP learning rate.
    stdp_tau_range : tuple of float
        Limits of both STDP time constants.
    stp_u_range : tuple of float
        Limits of the baseline release probability.
    homeostatic_target_range : tuple of float
        Limits of the target rate.
    bitstream_length_range : tuple of int
        Limits of the bitstream length.
    """

    stdp_lr_range: Tuple[float, float] = (1e-6, 0.1)
    stdp_tau_range: Tuple[float, float] = (1.0, 100.0)
    stp_u_range: Tuple[float, float] = (0.01, 0.99)
    homeostatic_target_range: Tuple[float, float] = (0.1, 100.0)
    bitstream_length_range: Tuple[int, int] = (32, 4096)

    def enforce(self, rules: PlasticityRuleSet) -> PlasticityRuleSet:
        """Clamp all parameters to valid ranges.

        Clamped: the learning rate, both time constants, the release
        probability, the target rate and the length; the amplitudes are
        floored at 1e-6. The STP time constants and the gain adaptation rate
        are not limited.

        Parameters
        ----------
        rules : PlasticityRuleSet
            The set to clamp, in place.

        Returns
        -------
        PlasticityRuleSet
            The same object.
        """
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
        """Check if all parameters are within constraints.

        Only the learning rate, ``tau_plus`` and the release probability are
        checked; ``tau_minus``, the target rate and the length are not, so a
        set :meth:`enforce` would change can still be valid.
        """
        lr = rules.stdp.lr
        if not (self.stdp_lr_range[0] <= lr <= self.stdp_lr_range[1]):
            return False
        tau = rules.stdp.tau_plus
        if not (self.stdp_tau_range[0] <= tau <= self.stdp_tau_range[1]):
            return False
        u = rules.stp.u_base
        return self.stp_u_range[0] <= u <= self.stp_u_range[1]
