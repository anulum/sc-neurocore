# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Evolutionary substrate engine-binding contracts

"""Public contracts for the evolutionary-substrate PyO3 functions."""

from __future__ import annotations

import sc_neurocore_engine as engine


def test_exported_function_names_are_stable() -> None:
    expected = (
        "py_evo_batch_mutate",
        "py_evo_batch_fitness",
        "py_evo_batch_crossover",
        "py_evo_diversity",
        "py_evo_novelty",
        "py_evo_tournament",
    )
    assert tuple(getattr(engine, name).__name__ for name in expected) == expected


def test_batch_mutation_is_seeded_and_does_not_mutate_python_input() -> None:
    population = [[0.0] * 8 for _ in range(6)]
    first = engine.py_evo_batch_mutate(population, 1.0, 0.1, 42)
    second = engine.py_evo_batch_mutate(population, 1.0, 0.1, 42)

    assert first == second
    assert any(weight != 0.0 for genome in first for weight in genome)
    assert population == [[0.0] * 8 for _ in range(6)]


def test_batch_fitness_preserves_score_ordering() -> None:
    fitness = engine.py_evo_batch_fitness(
        [[1.0, 0.0], [0.0, 1.0]],
        [1.0, 0.0],
        1.0,
    )
    assert fitness[0] == 0.0
    assert fitness[1] < fitness[0]


def test_batch_crossover_is_seeded_and_mixes_parent_genes() -> None:
    parents_a = [[0.0] * 32]
    parents_b = [[1.0] * 32]
    first = engine.py_evo_batch_crossover(parents_a, parents_b, 42)
    second = engine.py_evo_batch_crossover(parents_a, parents_b, 42)

    assert first == second
    assert len(first) == 1
    assert set(first[0]) == {0.0, 1.0}


def test_population_diversity_distinguishes_distinct_genomes() -> None:
    assert engine.py_evo_diversity([[1.0, 1.0], [1.0, 1.0]]) == 0.0
    assert engine.py_evo_diversity([[0.0, 0.0], [1.0, 1.0]]) > 0.0


def test_novelty_increases_with_archive_distance() -> None:
    scores = engine.py_evo_novelty(
        [[0.0, 0.0], [8.0, 8.0]],
        [[0.0, 0.0]],
        1,
    )
    assert scores[1] > scores[0]


def test_tournament_selection_is_seeded_and_returns_valid_indices() -> None:
    first = engine.py_evo_tournament([0.1, 0.9, 0.4], 12, 3, 42)
    second = engine.py_evo_tournament([0.1, 0.9, 0.4], 12, 3, 42)

    assert first == second
    assert len(first) == 12
    assert set(first) <= {0, 1, 2}
