# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestEvolutionaryNAS from former test_sc_nas_engine.py

"""Focused suite: TestEvolutionaryNAS from former test_sc_nas_engine.py."""

from __future__ import annotations

from sc_nas_engine_support import *  # noqa: F403

class TestEvolutionaryNAS:
    def test_search_returns_non_empty_front(self) -> None:
        report = run_nas(population_size=10, num_generations=5, seed=42)
        assert len(report.pareto_front) > 0

    def test_search_history_recorded(self) -> None:
        report = run_nas(population_size=10, num_generations=5, seed=42)
        assert len(report.search_history) == 5

    def test_best_accuracy_positive(self) -> None:
        report = run_nas(population_size=10, num_generations=10, seed=42)
        assert report.best_accuracy > 0.0

    def test_most_efficient_exists(self) -> None:
        report = run_nas(population_size=10, num_generations=5, seed=42)
        assert report.most_efficient is not None

    def test_budget_constraint_respected(self) -> None:
        budget = FPGAResourceBudget(max_luts=10_000_000)
        report = run_nas(budget=budget, population_size=10, num_generations=5, seed=42)
        for c in report.pareto_front:
            assert c.total_luts <= budget.max_luts

    def test_search_can_score_candidates_with_surrogate_optimizer(self) -> None:
        surrogate = _FakeSurrogateOptimiser()
        report = run_nas(
            population_size=6,
            num_generations=2,
            seed=42,
            surrogate_optimizer=surrogate,
        )

        assert surrogate.calls
        assert len(report.pareto_front) > 0
        assert all(candidate.accuracy == 0.99 for candidate in report.pareto_front)

    def test_summary_format(self) -> None:
        report = run_nas(population_size=10, num_generations=3, seed=42)
        s = report.summary()
        assert "SC-NAS Report" in s
        assert "Pareto front size" in s

    def test_wall_time_recorded(self) -> None:
        report = run_nas(population_size=10, num_generations=3, seed=42)
        assert report.wall_time_s > 0.0

    def test_mutation_preserves_min_layers(self) -> None:
        nas = EvolutionaryNAS(NASObjective(), FPGAResourceBudget(), seed=42)
        for _ in range(50):
            parent = nas._random_candidate()
            child = nas._mutate(parent, 1)
            assert len(child.layers) >= 2

    def test_crossover_produces_valid_candidate(self) -> None:
        nas = EvolutionaryNAS(NASObjective(), FPGAResourceBudget(), seed=42)
        a = nas._random_candidate()
        b = nas._random_candidate()
        child = nas._crossover(a, b, 1)
        assert len(child.layers) >= 1
        child.evaluate_resources()
        assert child.total_luts > 0

    def test_convergence_early_stop(self) -> None:
        report = run_nas(
            population_size=10,
            num_generations=100,
            seed=42,
            convergence_patience=3,
        )
        assert len(report.search_history) < 100

    def test_history_has_dsp_bram(self) -> None:
        report = run_nas(population_size=10, num_generations=3, seed=42)
        assert "best_dsp" in report.search_history[0]
        assert "best_bram_kb" in report.search_history[0]

    def test_neuron_count_mutation(self) -> None:
        nas = EvolutionaryNAS(NASObjective(), FPGAResourceBudget(), seed=42)
        parent = nas._random_candidate()
        original_neurons = [l.neurons for l in parent.layers]
        mutated_any = False
        for _ in range(50):
            child = nas._mutate(parent, 1)
            if any(
                c.neurons != o
                for c, o in zip(child.layers, original_neurons)
                if len(child.layers) == len(parent.layers)
            ):
                mutated_any = True
                break
        # With 50 tries and 1/6 chance of neuron_count mutation, extremely likely
        assert mutated_any
