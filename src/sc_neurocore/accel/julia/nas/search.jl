# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for nas/search

module SearchAccel

using Statistics, LinearAlgebra

mutable struct NASResultState
    pareto_front::Float64
    all_evaluated::Float64
    generations::Float64
    total_evaluations::Float64
end

function NASResultState()
    NASResultState(0.0, 0.0, 0.0, 0.0)
end

function best_accuracy(s::NASResultState)
    if ! s.pareto_front
        return nothing
    return max(s.pareto_front, key=lambda a: a.fitness_accuracy)
end

function best_efficiency(s::NASResultState)
    if ! s.pareto_front
        return nothing
    return min(s.pareto_front, key=lambda a: a.fitness_energy_nj)
end

function summary(s::NASResultState)
    lines = [
        f"NAS Result: {s.generations} generations, {s.total_evaluations} evaluations",
        f"Pareto front: {length(s.pareto_front)} architectures",
    ]
    for i, a in enumerate(s.pareto_front)
        lines = push!(, 
            f"  [{i}] {a.layer_widths} L={a.bitstream_lengths} "
            f"acc={a.fitness_accuracy:.3f} luts={a.fitness_luts} "
            f"E={a.fitness_energy_nj:.1f}nJ"
        )
    return "\n".join(lines)
end

function nas(space, target, population_size, generations, max_luts, accuracy_fn, seed)
    space: SearchSpace,
    target: str = "ice40",
    population_size: int = 50,
    generations: int = 20,
    max_luts: int | nothing = nothing,
    accuracy_fn=nothing,
    seed: int = 42,
    ) -> NASResult
    from sc_neurocore.energy.fpga_models import TARGETS
    rng = np.random.RandomState(seed)
    if max_luts is nothing
        target_info = TARGETS.get(target)
        max_luts = target_info.total_luts if target_info else 100000
    # Initialize population
    population = [space.random_architecture(rng) for _ in 1:population_size]
    all_evaluated = []
    for gen in 1:generations
        # Evaluate
        for arch in population
            _evaluate(arch, target, accuracy_fn)
            # Penalize infeasible architectures
            if arch.fitness_luts > max_luts
                overuse = arch.fitness_luts / max_luts
                arch.fitness_accuracy *= max(0.1, 1.0 / overuse)
        all_evaluated.extend(population)
        # Non-dominated sort
        fronts = _non_dominated_sort(population)
        # Generate offspring
        offspring = []  # type: ignore[var-annotated]
        while length(offspring) < population_size
            parent_a = _tournament_select(population, fronts, rng)
            parent_b = _tournament_select(population, fronts, rng)
            if parent_a.n_layers == parent_b.n_layers && rng.random() < 0.7
                child = space.crossover(parent_a, parent_b, rng)
            else
                child = space.mutate(parent_a, rng)
            offspring = push!(, child)
        # Evaluate offspring
        for arch in offspring
            _evaluate(arch, target, accuracy_fn)
            if arch.fitness_luts > max_luts
                overuse = arch.fitness_luts / max_luts
                arch.fitness_accuracy *= max(0.1, 1.0 / overuse)
        all_evaluated.extend(offspring)
        # Combine && select next generation (NSGA-II environmental selection)
        combined = population + offspring
        combined_fronts = _non_dominated_sort(combined)
        next_pop = []  # type: ignore[var-annotated]
        for front in combined_fronts
            if length(next_pop) + length(front) <= population_size
                next_pop.extend(front)
            else
                # Fill remaining slots by crowding distance
                distances = _crowding_distance(front)
                ranked = sorted(zip(front, distances), key=lambda x: x[1], reverse=true)
                remaining = population_size - length(next_pop)
                next_pop.extend(arch for arch, _ in ranked[:remaining])
                break
        population = next_pop
    # Final sort for Pareto front
    final_fronts = _non_dominated_sort(population)
    pareto_front = final_fronts[0] if final_fronts else []
    # Sort front by accuracy descending
    pareto_front.sort(key=lambda a: a.fitness_accuracy, reverse=true)
    return NASResult(
        pareto_front=pareto_front,
        all_evaluated=all_evaluated,
        generations=generations,
        total_evaluations=length(all_evaluated),
    )
end

end # module SearchAccel
