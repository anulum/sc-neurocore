# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SC Optimizer

"""Automated stochastic optimization toolkit.

Optimizes bitstream lengths, decorrelation strategy, and compute modes
to maximize accuracy within strict FPGA resource and power budgets.

Supports greedy knapsack and simulated annealing search strategies,
mixed-precision SC/deterministic hybrid policies, and Pareto frontier
extraction for multi-objective trade-off analysis.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

try:
    from sc_neurocore_engine import py_opt_sa_search, py_opt_extract_pareto

    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False


class DecorrelationStrategy(Enum):
    NONE = "None"
    LFSR = "LFSR"
    SOBOL = "Sobol"
    HALTON = "Halton"
    SCC_DECORRELATOR = "SCC_Decorrelator"


class ComputeMode(Enum):
    SC = "SC"
    DETERMINISTIC = "Deterministic"
    HYBRID = "Hybrid"


@dataclass
class HardwareBudget:
    max_luts: int
    max_power_mw: float
    max_latency_cycles: int = 0


@dataclass
class LayerProfile:
    id: str
    mac_count: int
    is_critical_path: bool = False


@dataclass
class LayerConfig:
    bitstream_length: int
    decorrelator: str
    mode: str
    luts_used: int
    power_used: float
    accuracy_score: float
    latency_cycles: int = 0


@dataclass
class OptimizerReport:
    config: Dict[str, LayerConfig]
    total_luts: int
    total_power_mw: float
    total_latency_cycles: int
    mean_accuracy: float
    pareto_frontier: List[Tuple[int, float, float]] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"LUTs: {self.total_luts}, Power: {self.total_power_mw:.2f} mW, "
            f"Latency: {self.total_latency_cycles} cycles, "
            f"Accuracy: {self.mean_accuracy:.4f}",
        ]
        for lid, cfg in self.config.items():
            lines.append(
                f"  {lid}: N={cfg.bitstream_length}, "
                f"decorr={cfg.decorrelator}, mode={cfg.mode}, "
                f"acc={cfg.accuracy_score:.4f}"
            )
        return "\n".join(lines)


class SCOptimizer:
    def __init__(self, budget: HardwareBudget):
        self.budget = budget
        self.bitstream_options = [64, 128, 256, 512, 1024, 2048]
        self.decorrelators = [s.value for s in DecorrelationStrategy]
        self.modes = [m.value for m in ComputeMode]

    def _estimate_resources(
        self,
        mac_count: int,
        length: int,
        decorr: str,
        mode: str,
    ) -> Tuple[int, float, float, int]:
        """Returns (LUTs, Power_mW, Accuracy_score, Latency_cycles)."""
        if mode == "Deterministic":
            luts = mac_count * 120
            power = mac_count * 0.5
            return luts, power, 1.0, 1

        if mode == "Hybrid":
            sc_frac = 0.7
            det_frac = 0.3
            sc_luts = int(mac_count * sc_frac) * 2 + int(math.log2(length)) * 5
            det_luts = int(mac_count * det_frac) * 120
            luts = sc_luts + det_luts
            power = mac_count * sc_frac * 0.01 * (length / 256) + mac_count * det_frac * 0.5
            accuracy = 0.95  # hybrid baseline
            latency = length
            if decorr == "Sobol":
                luts += int(mac_count * sc_frac * 15)
                accuracy = 0.97
            elif decorr == "LFSR":
                luts += 16
                accuracy = 0.96
            return luts, power, min(1.0, accuracy), latency

        # SC mode
        luts = mac_count * 2 + int(math.log2(length)) * 5
        power = mac_count * 0.01 * (length / 256)
        latency = length

        if decorr == "Sobol":
            luts += mac_count * 15
            accuracy = 1.0 - (1.0 / length)
        elif decorr == "Halton":
            luts += mac_count * 12
            accuracy = 1.0 - (1.2 / length)
        elif decorr == "SCC_Decorrelator":
            luts += mac_count * 8
            accuracy = 1.0 - (1.5 / length)
        elif decorr == "LFSR":
            luts += 16
            accuracy = 1.0 - (1.0 / math.sqrt(length))
        else:
            accuracy = 1.0 - (2.0 / math.sqrt(length))

        accuracy = max(0.1, min(1.0, accuracy))
        return luts, power, accuracy, latency

    def _generate_candidates(self, layer: LayerProfile) -> List[LayerConfig]:
        candidates = []
        for mode in self.modes:
            if mode == "Deterministic":
                l, p, a, lat = self._estimate_resources(layer.mac_count, 1, "None", mode)
                candidates.append(LayerConfig(1, "None", mode, l, p, a, lat))
                continue

            for length in self.bitstream_options:
                for decorr in self.decorrelators:
                    l, p, a, lat = self._estimate_resources(layer.mac_count, length, decorr, mode)
                    candidates.append(LayerConfig(length, decorr, mode, l, p, a, lat))
        return candidates

    def _is_feasible(self, config: Dict[str, LayerConfig]) -> bool:
        total_luts = sum(c.luts_used for c in config.values())
        total_power = sum(c.power_used for c in config.values())
        total_latency = max((c.latency_cycles for c in config.values()), default=0)
        if total_luts > self.budget.max_luts:
            return False
        if total_power > self.budget.max_power_mw:
            return False
        return not (
            self.budget.max_latency_cycles > 0 and total_latency > self.budget.max_latency_cycles
        )

    def _score(self, config: Dict[str, LayerConfig], network: List[LayerProfile]) -> float:
        total = 0.0
        weight_sum = 0.0
        for layer in network:
            w = 2.0 if layer.is_critical_path else 1.0
            total += config[layer.id].accuracy_score * w
            weight_sum += w
        return total / weight_sum if weight_sum > 0 else 0.0

    def _build_report(
        self,
        config: Dict[str, LayerConfig],
        network: List[LayerProfile],
        pareto: List[Tuple[int, float, float]] | None = None,
    ) -> OptimizerReport:
        total_luts = sum(c.luts_used for c in config.values())
        total_power = sum(c.power_used for c in config.values())
        total_latency = max((c.latency_cycles for c in config.values()), default=0)
        mean_acc = self._score(config, network)
        return OptimizerReport(
            config=config,
            total_luts=total_luts,
            total_power_mw=total_power,
            total_latency_cycles=total_latency,
            mean_accuracy=mean_acc,
            pareto_frontier=pareto or [],
        )

    # ------------------------------------------------------------------
    # Greedy search
    # ------------------------------------------------------------------

    def optimize(self, network: List[LayerProfile]) -> Optional[OptimizerReport]:
        """Greedy knapsack optimization maximizing weighted accuracy."""
        current_config: Dict[str, LayerConfig] = {}
        candidates_per_layer = {layer.id: self._generate_candidates(layer) for layer in network}

        for layer in network:
            cheapest = min(candidates_per_layer[layer.id], key=lambda c: c.luts_used)
            current_config[layer.id] = cheapest

        if not self._is_feasible(current_config):
            return None

        upgraded = True
        while upgraded:
            upgraded = False
            best_upgrade = None
            best_layer_id = None
            max_efficiency = 0.0

            for layer in network:
                curr = current_config[layer.id]
                for cand in candidates_per_layer[layer.id]:
                    if cand.accuracy_score <= curr.accuracy_score:
                        continue
                    trial = dict(current_config)
                    trial[layer.id] = cand
                    if not self._is_feasible(trial):
                        continue
                    lut_diff = cand.luts_used - curr.luts_used
                    score_gain = cand.accuracy_score - curr.accuracy_score
                    if layer.is_critical_path:
                        score_gain *= 2.0
                    eff = score_gain / lut_diff if lut_diff > 0 else float("inf")
                    if eff > max_efficiency:
                        max_efficiency = eff
                        best_upgrade = cand
                        best_layer_id = layer.id

            if best_upgrade and best_layer_id is not None:
                current_config[best_layer_id] = best_upgrade
                upgraded = True

        return self._build_report(current_config, network)

    # ------------------------------------------------------------------
    # Simulated annealing search
    # ------------------------------------------------------------------

    def optimize_annealing(
        self,
        network: List[LayerProfile],
        *,
        t_init: float = 1.0,
        t_min: float = 0.001,
        alpha: float = 0.95,
        max_iter: int = 2000,
        seed: int = 42,
    ) -> Optional[OptimizerReport]:
        """Simulated annealing for larger design spaces.

        Delegates to Rust engine when available (1000×+ speedup).
        Falls back to pure-Python implementation otherwise.
        """
        if _HAS_RUST:
            return self._optimize_annealing_rust(
                network,
                t_init=t_init,
                t_min=t_min,
                alpha=alpha,
                max_iter=max_iter,
                seed=seed,
            )
        return self._optimize_annealing_python(
            network,
            t_init=t_init,
            t_min=t_min,
            alpha=alpha,
            max_iter=max_iter,
            seed=seed,
        )

    def _optimize_annealing_rust(
        self,
        network: List[LayerProfile],
        *,
        t_init: float = 1.0,
        t_min: float = 0.001,
        alpha: float = 0.95,
        max_iter: int = 2000,
        seed: int = 42,
    ) -> Optional[OptimizerReport]:
        """Rust-accelerated SA path."""
        mac_counts = [layer.mac_count for layer in network]
        weights = [2.0 if layer.is_critical_path else 1.0 for layer in network]

        result = py_opt_sa_search(
            mac_counts,
            weights,
            self.budget.max_luts,
            self.budget.max_power_mw,
            self.budget.max_latency_cycles,
            t_init,
            t_min,
            alpha,
            max_iter,
            seed,
        )

        if not result.get("feasible", False):
            return None

        layer_luts = result["layer_luts"]
        layer_power = result["layer_power"]
        layer_accuracy = result["layer_accuracy"]

        config: Dict[str, LayerConfig] = {}
        for i, layer in enumerate(network):
            config[layer.id] = LayerConfig(
                bitstream_length=0,
                decorrelator="auto",
                mode="auto",
                luts_used=layer_luts[i],
                power_used=layer_power[i],
                accuracy_score=layer_accuracy[i],
            )

        pareto_luts = result.get("pareto_luts", [])
        pareto_power = result.get("pareto_power", [])
        pareto_score = result.get("pareto_score", [])

        if pareto_luts:
            pareto_result = py_opt_extract_pareto(
                pareto_luts,
                pareto_power,
                pareto_score,
            )
            frontier = self._sort_and_dedupe_frontier(
                list(
                    zip(
                        pareto_result["luts"],
                        pareto_result["power"],
                        pareto_result["score"],
                    )
                )
            )
        else:
            frontier = []

        return self._build_report(config, network, frontier)

    def _optimize_annealing_python(
        self,
        network: List[LayerProfile],
        *,
        t_init: float = 1.0,
        t_min: float = 0.001,
        alpha: float = 0.95,
        max_iter: int = 2000,
        seed: int = 42,
    ) -> Optional[OptimizerReport]:
        """Pure-Python SA fallback."""
        rng = random.Random(seed)
        candidates_per_layer = {layer.id: self._generate_candidates(layer) for layer in network}

        current: Dict[str, LayerConfig] = {}
        for layer in network:
            cheapest = min(candidates_per_layer[layer.id], key=lambda c: c.luts_used)
            current[layer.id] = cheapest

        if not self._is_feasible(current):
            return None

        best = dict(current)
        best_score = self._score(best, network)
        current_score = best_score

        t = t_init
        pareto_points: List[Tuple[int, float, float]] = []

        while t > t_min and max_iter > 0:
            max_iter -= 1
            layer = rng.choice(network)
            cand = rng.choice(candidates_per_layer[layer.id])
            trial = dict(current)
            trial[layer.id] = cand

            if not self._is_feasible(trial):
                t *= alpha
                continue

            trial_score = self._score(trial, network)
            delta = trial_score - current_score

            if delta > 0 or rng.random() < math.exp(delta / t):
                current = trial
                current_score = trial_score
                if current_score > best_score:
                    best = dict(current)
                    best_score = current_score
                luts = sum(c.luts_used for c in current.values())
                power = sum(c.power_used for c in current.values())
                pareto_points.append((luts, power, current_score))

            t *= alpha

        frontier = self._extract_pareto(pareto_points)
        return self._build_report(best, network, frontier)

    @staticmethod
    def _extract_pareto(
        points: List[Tuple[int, float, float]],
    ) -> List[Tuple[int, float, float]]:
        """Extract non-dominated Pareto frontier from (luts, power, accuracy) tuples."""
        if not points:
            return []
        frontier = []
        for p in points:
            dominated = False
            for q in points:
                if q is p:
                    continue
                # q dominates p if q uses ≤ resources AND has ≥ accuracy
                if q[0] <= p[0] and q[1] <= p[1] and q[2] >= p[2]:
                    if q[0] < p[0] or q[1] < p[1] or q[2] > p[2]:
                        dominated = True
                        break
            if not dominated:
                frontier.append(p)
        return SCOptimizer._sort_and_dedupe_frontier(frontier)

    @staticmethod
    def _sort_and_dedupe_frontier(
        frontier: List[Tuple[int, float, float]],
    ) -> List[Tuple[int, float, float]]:
        """Return the frontier sorted by LUTs ascending with duplicate points removed.

        Both the Rust-accelerated and pure-Python annealing paths funnel their
        non-dominated points through this normaliser, so the reported Pareto
        frontier honours the same ordering and deduplication contract regardless
        of which backend produced it.
        """
        seen: set[Tuple[int, float, float]] = set()
        deduped: List[Tuple[int, float, float]] = []
        for pt in sorted(frontier, key=lambda x: x[0]):
            key = (pt[0], round(pt[1], 4), round(pt[2], 4))
            if key not in seen:
                seen.add(key)
                deduped.append(pt)
        return deduped
