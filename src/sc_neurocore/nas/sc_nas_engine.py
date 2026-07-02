# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Hardware-Aware SC-NAS Engine (Evolutionary)

"""Evolutionary neural architecture search for SC bitstream hardware.

Jointly optimises topology, neuron types, per-layer bitstream lengths,
and decorrelation strategies against an FPGA resource budget.  Produces
Pareto-optimal SC networks with auto-generated SystemVerilog via the
model zoo ``VerilogGenerator``.

No external dependencies beyond NumPy — the evaluator uses pure-Python
SC simulation (bitstream variance model), so ``torch`` is NOT required.
"""

from __future__ import annotations

import copy
import hashlib
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

try:
    from sc_neurocore_engine import py_evo_tournament

    _HAS_RUST_EVO = True
except ImportError:
    _HAS_RUST_EVO = False


class DecorrelationStrategy(Enum):
    """Supported bitstream decorrelation generators for SC-NAS candidates."""

    LFSR = "lfsr"
    SOBOL = "sobol"
    HALTON = "halton"
    HYBRID = "hybrid"


class NeuronType(Enum):
    """Neuron model families available to the hardware-aware NAS search."""

    LIF = "LIF"
    IZHIKEVICH = "Izhikevich"
    ADEX = "AdEx"
    HH = "Hodgkin-Huxley"


# Neuron-type hardware complexity multipliers (vs LIF baseline)
NEURON_LUT_MULTIPLIER: Dict[NeuronType, float] = {
    NeuronType.LIF: 1.0,
    NeuronType.IZHIKEVICH: 1.8,
    NeuronType.ADEX: 2.2,
    NeuronType.HH: 4.5,
}
NEURON_DSP_COST: Dict[NeuronType, int] = {
    NeuronType.LIF: 0,
    NeuronType.IZHIKEVICH: 1,
    NeuronType.ADEX: 2,
    NeuronType.HH: 4,
}


@dataclass
class FPGAResourceBudget:
    """Hardware resource constraints for the target FPGA."""

    max_luts: int = 500_000
    max_ffs: int = 500_000
    max_bram_kb: int = 2048
    max_dsp: int = 256
    max_power_mw: float = 5000.0

    def utilisation(self, luts: int, ffs: int, bram: int, dsp: int) -> Dict[str, float]:
        """Return per-resource utilisation ratios for a candidate design."""
        return {
            "luts": luts / self.max_luts,
            "ffs": ffs / self.max_ffs,
            "bram": bram / self.max_bram_kb,
            "dsp": dsp / self.max_dsp,
        }


@dataclass
class NASObjective:
    """Search objectives and constraints."""

    min_accuracy: float = 0.90
    min_bitstream_length: int = 64
    max_bitstream_length: int = 4096
    allowed_neuron_types: List[NeuronType] = field(default_factory=lambda: list(NeuronType))
    allowed_decorrelators: List[DecorrelationStrategy] = field(
        default_factory=lambda: list(DecorrelationStrategy)
    )


@dataclass
class LayerConfig:
    """Configuration for a single network layer."""

    neurons: int
    neuron_type: NeuronType
    bitstream_length: int
    decorrelation: DecorrelationStrategy

    @property
    def lut_cost(self) -> int:
        """Return estimated LUT cost for this layer."""
        base = self.neurons * 12
        length_factor = int(math.log2(max(64, self.bitstream_length))) * 5
        type_mult = NEURON_LUT_MULTIPLIER.get(self.neuron_type, 1.0)
        return int((base + length_factor * self.neurons) * type_mult)

    @property
    def ff_cost(self) -> int:
        """Return estimated flip-flop cost for this layer."""
        return self.neurons * (self.bitstream_length // 64 + 8)

    @property
    def dsp_cost(self) -> int:
        """Return estimated DSP block cost for this layer."""
        per_neuron = NEURON_DSP_COST.get(self.neuron_type, 0)
        return self.neurons * per_neuron

    @property
    def bram_cost_kb(self) -> float:
        """Return estimated BRAM storage cost in kibibytes."""
        # Weight storage: neurons × bitstream_length bits → KB
        return (self.neurons * self.bitstream_length) / 8192.0

    @property
    def power_cost(self) -> float:
        """Return estimated dynamic power cost in milliwatts."""
        type_mult = NEURON_LUT_MULTIPLIER.get(self.neuron_type, 1.0)
        return self.neurons * 0.01 * (self.bitstream_length / 256.0) * type_mult


@dataclass
class SCCandidate:
    """A candidate SC network architecture."""

    layers: List[LayerConfig]
    fitness: float = 0.0
    accuracy: float = 0.0
    total_luts: int = 0
    total_ffs: int = 0
    total_dsp: int = 0
    total_bram_kb: float = 0.0
    total_power_mw: float = 0.0
    generation: int = 0
    crowding_distance: float = 0.0

    def evaluate_resources(self) -> None:
        """Update aggregate resource estimates from the candidate layers."""
        self.total_luts = sum(l.lut_cost for l in self.layers)
        self.total_ffs = sum(l.ff_cost for l in self.layers)
        self.total_dsp = sum(l.dsp_cost for l in self.layers)
        self.total_bram_kb = sum(l.bram_cost_kb for l in self.layers)
        self.total_power_mw = sum(l.power_cost for l in self.layers)

    def meets_budget(self, budget: FPGAResourceBudget) -> bool:
        """Return whether this candidate fits within an FPGA resource budget."""
        self.evaluate_resources()
        return (
            self.total_luts <= budget.max_luts
            and self.total_ffs <= budget.max_ffs
            and self.total_dsp <= budget.max_dsp
            and self.total_bram_kb <= budget.max_bram_kb
            and self.total_power_mw <= budget.max_power_mw
        )

    @property
    def fingerprint(self) -> str:
        """Return a deterministic non-cryptographic architecture fingerprint."""
        desc = "|".join(
            f"{l.neurons}-{l.neuron_type.value}-{l.bitstream_length}-{l.decorrelation.value}"
            for l in self.layers
        )
        # MD5 used for de-duplication of architecture descriptors —
        # NOT a security boundary. `usedforsecurity=False` tells
        # bandit/B324 + FIPS-140 that this is a non-cryptographic use.
        return hashlib.md5(desc.encode(), usedforsecurity=False).hexdigest()[:12]


# ── Fitness Evaluator ────────────────────────────────────────────────


class SCFitnessEvaluator:
    """Pure-Python SC simulation fitness evaluator.

    Uses the SC variance model: for a bitstream of length N encoding
    probability p, the variance is p*(1-p)/N.  Accuracy is estimated
    as 1 − mean_variance across all layers.
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def evaluate(self, candidate: SCCandidate, target_p: float = 0.5) -> float:
        """Evaluate candidate accuracy via SC variance model."""
        variances = []
        for layer in candidate.layers:
            p = target_p
            var = p * (1 - p) / layer.bitstream_length
            decorr_bonus = {
                DecorrelationStrategy.LFSR: 1.0,
                DecorrelationStrategy.SOBOL: 0.7,
                DecorrelationStrategy.HALTON: 0.8,
                DecorrelationStrategy.HYBRID: 0.6,
            }[layer.decorrelation]
            variances.append(var * decorr_bonus)
        mean_var = float(np.mean(variances)) if variances else 0.5
        accuracy = max(0.0, min(1.0, 1.0 - mean_var * 10.0))
        candidate.accuracy = accuracy
        return accuracy


# ── Pareto Front ─────────────────────────────────────────────────────


def pareto_front(
    candidates: List[SCCandidate],
    objectives: Sequence[str] = ("accuracy", "total_luts"),
) -> List[SCCandidate]:
    """Extract the Pareto-optimal front (NSGA-II non-dominated sorting).

    Maximises accuracy, minimises resource usage.
    """
    if not candidates:
        return []

    def dominates(a: SCCandidate, b: SCCandidate) -> bool:
        a_vals = (a.accuracy, -a.total_luts, -a.total_power_mw)
        b_vals = (b.accuracy, -b.total_luts, -b.total_power_mw)
        better_in_any = False
        for av, bv in zip(a_vals, b_vals):
            if av < bv:
                return False
            if av > bv:
                better_in_any = True
        return better_in_any

    front = []
    for c in candidates:
        dominated = False
        for other in candidates:
            if other is not c and dominates(other, c):
                dominated = True
                break
        if not dominated:
            front.append(c)

    # Compute crowding distance for diversity
    if len(front) >= 3:
        _assign_crowding_distance(front)

    return front


def _assign_crowding_distance(front: List[SCCandidate]) -> None:
    """Assign NSGA-II crowding distance to Pareto front members."""
    n = len(front)
    for c in front:
        c.crowding_distance = 0.0

    for attr in ("accuracy", "total_luts", "total_power_mw"):
        front.sort(key=lambda c: getattr(c, attr))
        front[0].crowding_distance = float("inf")
        front[-1].crowding_distance = float("inf")
        obj_range = getattr(front[-1], attr) - getattr(front[0], attr)
        if obj_range == 0:
            continue
        for i in range(1, n - 1):
            diff = getattr(front[i + 1], attr) - getattr(front[i - 1], attr)
            front[i].crowding_distance += diff / obj_range


# ── Evolutionary NAS ─────────────────────────────────────────────────


class EvolutionaryNAS:
    """µ+λ evolutionary search with tournament selection."""

    def __init__(
        self,
        objective: NASObjective,
        budget: FPGAResourceBudget,
        population_size: int = 50,
        num_generations: int = 100,
        mutation_rate: float = 0.3,
        seed: int = 42,
        convergence_patience: int = 0,
        surrogate_optimizer: Any | None = None,
    ):
        self.objective = objective
        self.budget = budget
        self.pop_size = population_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.convergence_patience = convergence_patience
        self.rng = np.random.default_rng(seed)
        self.evaluator = SCFitnessEvaluator(seed)
        self.surrogate_optimizer = surrogate_optimizer
        self.history: List[Dict[str, Any]] = []

    def _random_layer(self) -> LayerConfig:
        neuron_types = self.objective.allowed_neuron_types
        decorrelators = self.objective.allowed_decorrelators
        return LayerConfig(
            neurons=int(self.rng.choice([16, 32, 64, 128, 256])),
            neuron_type=neuron_types[int(self.rng.integers(0, len(neuron_types)))],
            bitstream_length=int(self.rng.choice([64, 128, 256, 512, 1024, 2048, 4096])),
            decorrelation=decorrelators[int(self.rng.integers(0, len(decorrelators)))],
        )

    def _random_candidate(self, gen: int = 0) -> SCCandidate:
        n_layers = int(self.rng.integers(2, 6))
        layers = [self._random_layer() for _ in range(n_layers)]
        c = SCCandidate(layers=layers, generation=gen)
        c.evaluate_resources()
        return c

    def _mutate(self, candidate: SCCandidate, gen: int) -> SCCandidate:
        c = SCCandidate(
            layers=[copy.deepcopy(l) for l in candidate.layers],
            generation=gen,
        )
        action = self.rng.choice(["length", "neuron", "decorr", "add", "remove", "neuron_count"])

        if action == "length" and c.layers:
            idx = int(self.rng.integers(0, len(c.layers)))
            factor = self.rng.choice([0.5, 2.0])
            new_len = int(c.layers[idx].bitstream_length * factor)
            c.layers[idx].bitstream_length = max(
                self.objective.min_bitstream_length,
                min(self.objective.max_bitstream_length, new_len),
            )
        elif action == "neuron" and c.layers:
            idx = int(self.rng.integers(0, len(c.layers)))
            neuron_types = self.objective.allowed_neuron_types
            c.layers[idx].neuron_type = neuron_types[int(self.rng.integers(0, len(neuron_types)))]
        elif action == "decorr" and c.layers:
            idx = int(self.rng.integers(0, len(c.layers)))
            decorrelators = self.objective.allowed_decorrelators
            c.layers[idx].decorrelation = decorrelators[
                int(self.rng.integers(0, len(decorrelators)))
            ]
        elif action == "add":
            c.layers.append(self._random_layer())
        elif action == "remove" and len(c.layers) > 2:
            idx = int(self.rng.integers(0, len(c.layers)))
            c.layers.pop(idx)
        elif action == "neuron_count" and c.layers:
            idx = int(self.rng.integers(0, len(c.layers)))
            factor = self.rng.choice([0.5, 2.0])
            c.layers[idx].neurons = max(4, min(512, int(c.layers[idx].neurons * factor)))

        c.evaluate_resources()
        return c

    def _crossover(self, a: SCCandidate, b: SCCandidate, gen: int) -> SCCandidate:
        min_len = min(len(a.layers), len(b.layers))
        layers = []
        for i in range(min_len):
            layers.append(copy.deepcopy(a.layers[i] if self.rng.random() < 0.5 else b.layers[i]))
        c = SCCandidate(layers=layers, generation=gen)
        c.evaluate_resources()
        return c

    def _tournament_select(self, population: List[SCCandidate], k: int = 3) -> SCCandidate:
        if _HAS_RUST_EVO and len(population) > 20:
            fitness = [c.fitness for c in population]
            indices = py_evo_tournament(fitness, 1, k, int(self.rng.integers(0, 2**32)))
            return population[int(indices[0])]
        k_sel = min(k, len(population))
        sel_idx = self.rng.choice(len(population), size=k_sel, replace=False)
        candidates = [population[int(i)] for i in sel_idx]
        return max(candidates, key=lambda c: c.fitness)

    def _evaluate_candidate(self, candidate: SCCandidate) -> SCCandidate:
        """Evaluate a candidate through the configured NAS scoring path."""
        if self.surrogate_optimizer is not None:
            from sc_neurocore.nas.surrogate_bridge import evaluate_candidate_with_surrogate

            return evaluate_candidate_with_surrogate(
                candidate,
                self.surrogate_optimizer,
                budget=self.budget,
            ).candidate

        acc = self.evaluator.evaluate(candidate)
        penalty = 0.0 if candidate.meets_budget(self.budget) else 0.5
        candidate.fitness = acc - penalty
        return candidate

    def search(self) -> List[SCCandidate]:
        """Run the evolutionary search. Returns the final Pareto front."""
        population = [self._random_candidate(0) for _ in range(self.pop_size)]

        population = [self._evaluate_candidate(c) for c in population]

        stale_count = 0
        prev_best = -1.0

        for gen in range(1, self.num_generations + 1):
            offspring = []
            for _ in range(self.pop_size):
                if self.rng.random() < self.mutation_rate:
                    parent = self._tournament_select(population)
                    child = self._mutate(parent, gen)
                else:
                    p1 = self._tournament_select(population)
                    p2 = self._tournament_select(population)
                    child = self._crossover(p1, p2, gen)
                offspring.append(self._evaluate_candidate(child))

            combined = population + offspring
            combined.sort(key=lambda c: c.fitness, reverse=True)
            population = combined[: self.pop_size]

            best = population[0]
            self.history.append(
                {
                    "generation": gen,
                    "best_fitness": best.fitness,
                    "best_accuracy": best.accuracy,
                    "best_luts": best.total_luts,
                    "best_dsp": best.total_dsp,
                    "best_bram_kb": best.total_bram_kb,
                    "best_power": best.total_power_mw,
                    "pop_size": len(population),
                }
            )

            # Convergence detection
            if self.convergence_patience > 0:
                if abs(best.fitness - prev_best) < 1e-8:
                    stale_count += 1
                else:
                    stale_count = 0
                prev_best = best.fitness
                if stale_count >= self.convergence_patience:
                    break

        return pareto_front(population)


# ── NAS Report ───────────────────────────────────────────────────────


@dataclass
class NASReport:
    """Summary report from an SC-NAS search."""

    pareto_front: List[SCCandidate]
    search_history: List[Dict[str, Any]]
    wall_time_s: float = 0.0

    @property
    def best_accuracy(self) -> float:
        """Return the best accuracy in the Pareto front, or zero when empty."""
        if not self.pareto_front:
            return 0.0
        return max(c.accuracy for c in self.pareto_front)

    @property
    def most_efficient(self) -> Optional[SCCandidate]:
        """Return the lowest-LUT candidate in the Pareto front, if present."""
        if not self.pareto_front:
            return None
        return min(self.pareto_front, key=lambda c: c.total_luts)

    def summary(self) -> str:
        """Return a deterministic human-readable search summary."""
        lines = [
            "SC-NAS Report",
            f"  Pareto front size: {len(self.pareto_front)}",
            f"  Best accuracy: {self.best_accuracy:.4f}",
            f"  Search time: {self.wall_time_s:.2f}s",
        ]
        if self.most_efficient:
            e = self.most_efficient
            lines.append(f"  Most efficient: {e.total_luts} LUTs, {e.accuracy:.4f} acc")
        return "\n".join(lines)


def run_nas(
    objective: Optional[NASObjective] = None,
    budget: Optional[FPGAResourceBudget] = None,
    population_size: int = 50,
    num_generations: int = 100,
    seed: int = 42,
    convergence_patience: int = 0,
    surrogate_optimizer: Any | None = None,
) -> NASReport:
    """Run an SC-NAS search and return its report."""
    obj = objective or NASObjective()
    bgt = budget or FPGAResourceBudget()
    engine = EvolutionaryNAS(
        obj,
        bgt,
        population_size,
        num_generations,
        seed=seed,
        convergence_patience=convergence_patience,
        surrogate_optimizer=surrogate_optimizer,
    )
    t0 = time.perf_counter()
    front = engine.search()
    elapsed = time.perf_counter() - t0
    return NASReport(
        pareto_front=front,
        search_history=engine.history,
        wall_time_s=elapsed,
    )


# ── Verilog Emitter ─────────────────────────────────────────────────


class NASVerilogEmitter:
    """Emits SystemVerilog for Pareto-optimal SC-NAS candidates."""

    @staticmethod
    def emit(candidate: SCCandidate, module_name: str = "sc_nas_network") -> str:
        """Generate SystemVerilog for a searched architecture."""
        lines = [
            "// SC-NeuroCore — SC-NAS Auto-Generated Architecture",
            f"// Fingerprint: {candidate.fingerprint}",
            f"// Accuracy: {candidate.accuracy:.4f}",
            f"// Resources: {candidate.total_luts} LUTs, {candidate.total_dsp} DSPs, "
            f"{candidate.total_bram_kb:.1f} KB BRAM, {candidate.total_power_mw:.2f} mW",
            "",
            f"module {module_name} #(",
        ]

        params = []
        for i, layer in enumerate(candidate.layers):
            params.append(f"    parameter L{i}_NEURONS    = {layer.neurons},")
            params.append(f"    parameter L{i}_BITSTREAM  = {layer.bitstream_length},")
            params.append(f'    parameter L{i}_DECORR     = "{layer.decorrelation.value}",')
        if params:
            params[-1] = params[-1].rstrip(",")
        lines.extend(params)

        lines.append(")(")
        lines.append("    input  logic clk,")
        lines.append("    input  logic rst_n,")

        n_in = candidate.layers[0].neurons if candidate.layers else 16
        n_out = candidate.layers[-1].neurons if candidate.layers else 16
        bs_in = candidate.layers[0].bitstream_length if candidate.layers else 256
        bs_out = candidate.layers[-1].bitstream_length if candidate.layers else 256
        lines.append(f"    input  logic [{bs_in - 1}:0] sc_input  [0:{n_in - 1}],")
        lines.append(f"    output logic [{bs_out - 1}:0] sc_output [0:{n_out - 1}],")
        lines.append(f"    output logic [{n_out - 1}:0] spike_out")
        lines.append(");")
        lines.append("")

        # Instantiate layers
        for i, layer in enumerate(candidate.layers):
            neuron_module = {
                NeuronType.LIF: "sc_lif_neuron",
                NeuronType.IZHIKEVICH: "sc_izhikevich_neuron",
                NeuronType.ADEX: "sc_adex_neuron",
                NeuronType.HH: "sc_hh_neuron",
            }.get(layer.neuron_type, "sc_lif_neuron")

            lines.append(
                f"    // Layer {i}: {layer.neurons} × {neuron_module} "
                f"(N={layer.bitstream_length}, {layer.decorrelation.value})"
            )
            lines.append(f"    genvar g{i};")
            lines.append("    generate")
            lines.append(
                f"        for (g{i} = 0; g{i} < L{i}_NEURONS; g{i} = g{i} + 1) begin : layer{i}_gen"
            )
            lines.append(f"            {neuron_module} #(")
            lines.append(f"                .BITSTREAM_W(L{i}_BITSTREAM)")
            lines.append(f"            ) u_l{i} (")
            lines.append("                .clk(clk),")
            lines.append("                .rst_n(rst_n)")
            lines.append("            );")
            lines.append("        end")
            lines.append("    endgenerate")
            lines.append("")

        lines.append("endmodule")
        return "\n".join(lines)

    @staticmethod
    def emit_pareto(front: List[SCCandidate]) -> Dict[str, str]:
        """Emit Verilog for all Pareto-optimal candidates."""
        result = {}
        for i, c in enumerate(front):
            name = f"sc_nas_pareto_{i}"
            result[name] = NASVerilogEmitter.emit(c, module_name=name)
        return result
