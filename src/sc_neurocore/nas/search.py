# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Hardware-aware SNN NAS engine

"""NSGA-II evolutionary search over SNN architectures under FPGA constraints.

Searches {neuron model, layer width, bitstream length, delay range} jointly,
evaluating each candidate for accuracy (simulated) and hardware cost (via
the energy estimator). Returns a Pareto front of non-dominated architectures.

No equivalent exists: SpikeNAS searches only software architectures.
This is the first NAS that searches hardware parameters (L, delays, LUTs)
alongside network topology.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sc_neurocore.energy.estimator import estimate

from .search_space import Architecture, SearchSpace


@dataclass
class NASResult:
    """Result of a NAS run."""

    pareto_front: list[Architecture]
    all_evaluated: list[Architecture]
    generations: int
    total_evaluations: int

    def best_accuracy(self) -> Architecture | None:
        """Architecture with highest accuracy on the Pareto front."""
        if not self.pareto_front:
            return None
        return max(self.pareto_front, key=lambda a: a.fitness_accuracy)

    def best_efficiency(self) -> Architecture | None:
        """Architecture with lowest energy on the Pareto front."""
        if not self.pareto_front:
            return None
        return min(self.pareto_front, key=lambda a: a.fitness_energy_nj)

    def summary(self) -> str:
        lines = [
            f"NAS Result: {self.generations} generations, {self.total_evaluations} evaluations",
            f"Pareto front: {len(self.pareto_front)} architectures",
        ]
        for i, a in enumerate(self.pareto_front):
            lines.append(
                f"  [{i}] {a.layer_widths} L={a.bitstream_lengths} "
                f"acc={a.fitness_accuracy:.3f} luts={a.fitness_luts} "
                f"E={a.fitness_energy_nj:.1f}nJ"
            )
        return "\n".join(lines)


def _evaluate(  # type: ignore[no-untyped-def]
    arch: Architecture,
    target: str,
    accuracy_fn=None,
) -> Architecture:
    """Evaluate one architecture: hardware cost + optional accuracy."""
    avg_L = int(np.mean(arch.bitstream_lengths))
    report = estimate(arch.layer_sizes, target=target, bitstream_length=avg_L)

    arch.fitness_luts = report.total_luts
    arch.fitness_energy_nj = report.energy_per_inference_nj

    if accuracy_fn is not None:
        arch.fitness_accuracy = accuracy_fn(arch)
    else:
        # Proxy: larger networks with longer bitstreams are more accurate
        param_score = min(arch.total_params / 10000, 1.0)
        L_score = min(avg_L / 256, 1.0)
        arch.fitness_accuracy = 0.5 * param_score + 0.5 * L_score

    return arch


def _dominates(a: Architecture, b: Architecture) -> bool:
    """True if a Pareto-dominates b (higher accuracy AND lower energy)."""
    better_acc = a.fitness_accuracy >= b.fitness_accuracy
    better_energy = a.fitness_energy_nj <= b.fitness_energy_nj
    strictly = a.fitness_accuracy > b.fitness_accuracy or a.fitness_energy_nj < b.fitness_energy_nj
    return better_acc and better_energy and strictly


def _non_dominated_sort(population: list[Architecture]) -> list[list[Architecture]]:
    """NSGA-II non-dominated sorting. Returns list of fronts."""
    n = len(population)
    domination_counts = [0] * n
    dominated_sets: list[list[int]] = [[] for _ in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            if _dominates(population[i], population[j]):
                dominated_sets[i].append(j)
                domination_counts[j] += 1
            elif _dominates(population[j], population[i]):
                dominated_sets[j].append(i)
                domination_counts[i] += 1

    fronts: list[list[Architecture]] = []
    current_front_indices = [i for i in range(n) if domination_counts[i] == 0]

    while current_front_indices:
        front = [population[i] for i in current_front_indices]
        fronts.append(front)
        next_front = []
        for i in current_front_indices:
            for j in dominated_sets[i]:
                domination_counts[j] -= 1
                if domination_counts[j] == 0:
                    next_front.append(j)
        current_front_indices = next_front

    return fronts


def _crowding_distance(front: list[Architecture]) -> list[float]:
    """Compute NSGA-II crowding distance for a front."""
    n = len(front)
    if n <= 2:
        return [float("inf")] * n

    distances = [0.0] * n

    for key in ("fitness_accuracy", "fitness_energy_nj"):
        indices = sorted(range(n), key=lambda i: getattr(front[i], key))
        obj_min = getattr(front[indices[0]], key)
        obj_max = getattr(front[indices[-1]], key)
        obj_range = obj_max - obj_min if obj_max != obj_min else 1.0

        distances[indices[0]] = float("inf")
        distances[indices[-1]] = float("inf")

        for k in range(1, n - 1):
            val_next = getattr(front[indices[k + 1]], key)
            val_prev = getattr(front[indices[k - 1]], key)
            distances[indices[k]] += (val_next - val_prev) / obj_range

    return distances


def _tournament_select(
    population: list[Architecture],
    fronts: list[list[Architecture]],
    rng: np.random.RandomState,
) -> Architecture:
    """Binary tournament selection using front rank + crowding distance."""
    # Build rank map
    rank_map = {}
    for rank, front in enumerate(fronts):
        for arch in front:
            rank_map[id(arch)] = rank

    i, j = rng.choice(len(population), size=2, replace=False)
    a, b = population[int(i)], population[int(j)]
    rank_a = rank_map.get(id(a), len(fronts))
    rank_b = rank_map.get(id(b), len(fronts))

    if rank_a < rank_b:
        return a
    if rank_b < rank_a:
        return b
    return a if rng.random() < 0.5 else b


def nas(  # type: ignore[no-untyped-def]
    space: SearchSpace,
    target: str = "ice40",
    population_size: int = 50,
    generations: int = 20,
    max_luts: int | None = None,
    accuracy_fn=None,
    seed: int = 42,
) -> NASResult:
    """Run hardware-aware NAS using NSGA-II.

    Parameters
    ----------
    space : SearchSpace
        Architecture search space definition.
    target : str
        FPGA target for hardware cost evaluation.
    population_size : int
        Number of architectures per generation.
    generations : int
        Number of evolutionary generations.
    max_luts : int, optional
        Hard LUT budget. Architectures exceeding this are penalized.
        If None, uses the target's total LUT count.
    accuracy_fn : callable, optional
        Function(Architecture) -> float accuracy in [0, 1].
        If None, uses a proxy based on network capacity.
    seed : int
        Random seed.

    Returns
    -------
    NASResult
        Pareto front + all evaluated architectures.
    """
    from sc_neurocore.energy.fpga_models import TARGETS

    rng = np.random.RandomState(seed)

    if max_luts is None:
        target_info = TARGETS.get(target)
        max_luts = target_info.total_luts if target_info else 100000

    # Initialize population
    population = [space.random_architecture(rng) for _ in range(population_size)]
    all_evaluated = []

    for gen in range(generations):
        # Evaluate
        for arch in population:
            _evaluate(arch, target, accuracy_fn)
            # Penalize infeasible architectures
            if arch.fitness_luts > max_luts:
                overuse = arch.fitness_luts / max_luts
                arch.fitness_accuracy *= max(0.1, 1.0 / overuse)

        all_evaluated.extend(population)

        # Non-dominated sort
        fronts = _non_dominated_sort(population)

        # Generate offspring
        offspring = []  # type: ignore[var-annotated]
        while len(offspring) < population_size:
            parent_a = _tournament_select(population, fronts, rng)
            parent_b = _tournament_select(population, fronts, rng)

            if parent_a.n_layers == parent_b.n_layers and rng.random() < 0.7:
                child = space.crossover(parent_a, parent_b, rng)
            else:
                child = space.mutate(parent_a, rng)

            offspring.append(child)

        # Evaluate offspring
        for arch in offspring:
            _evaluate(arch, target, accuracy_fn)
            if arch.fitness_luts > max_luts:
                overuse = arch.fitness_luts / max_luts
                arch.fitness_accuracy *= max(0.1, 1.0 / overuse)

        all_evaluated.extend(offspring)

        # Combine and select next generation (NSGA-II environmental selection)
        combined = population + offspring
        combined_fronts = _non_dominated_sort(combined)

        next_pop = []  # type: ignore[var-annotated]
        for front in combined_fronts:
            if len(next_pop) + len(front) <= population_size:
                next_pop.extend(front)
            else:
                # Fill remaining slots by crowding distance
                distances = _crowding_distance(front)
                ranked = sorted(zip(front, distances), key=lambda x: x[1], reverse=True)
                remaining = population_size - len(next_pop)
                next_pop.extend(arch for arch, _ in ranked[:remaining])
                break

        population = next_pop

    # Final sort for Pareto front
    final_fronts = _non_dominated_sort(population)
    pareto_front = final_fronts[0] if final_fronts else []

    # Sort front by accuracy descending
    pareto_front.sort(key=lambda a: a.fitness_accuracy, reverse=True)

    return NASResult(
        pareto_front=pareto_front,
        all_evaluated=all_evaluated,
        generations=generations,
        total_evaluations=len(all_evaluated),
    )
