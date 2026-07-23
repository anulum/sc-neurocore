# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestNAS from former test_nas.py

"""Focused suite: TestNAS from former test_nas.py."""

from __future__ import annotations

from tests.nas_support import *  # noqa: F403

class TestNAS:
    def test_small_search(self) -> None:
        sp = SearchSpace(n_inputs=16, n_outputs=4, min_layers=1, max_layers=2)
        result = nas(sp, target="artix7", population_size=10, generations=3, seed=42)
        assert isinstance(result, NASResult)
        assert len(result.pareto_front) > 0
        assert result.generations == 3
        assert result.total_evaluations > 0

    def test_best_accuracy(self) -> None:
        sp = SearchSpace(n_inputs=16, n_outputs=4, min_layers=1, max_layers=2)
        result = nas(sp, target="artix7", population_size=10, generations=3)
        best = result.best_accuracy()
        assert best is not None
        assert best.fitness_accuracy > 0

    def test_best_efficiency(self) -> None:
        sp = SearchSpace(n_inputs=16, n_outputs=4, min_layers=1, max_layers=2)
        result = nas(sp, target="artix7", population_size=10, generations=3)
        best = result.best_efficiency()
        assert best is not None
        assert best.fitness_energy_nj > 0

    def test_summary(self) -> None:
        sp = SearchSpace(n_inputs=16, n_outputs=4, min_layers=1, max_layers=2)
        result = nas(sp, target="ice40", population_size=10, generations=2)
        s = result.summary()
        assert "NAS Result" in s
        assert "Pareto front" in s

    def test_lut_constraint(self) -> None:
        sp = SearchSpace(n_inputs=16, n_outputs=4, min_layers=1, max_layers=2)
        result = nas(sp, target="ice40", population_size=10, generations=3, max_luts=5000)
        for arch in result.pareto_front:
            assert arch.fitness_luts > 0

    def test_custom_accuracy(self) -> None:
        sp = SearchSpace(n_inputs=8, n_outputs=2, min_layers=1, max_layers=1)
        result = nas(
            sp,
            target="artix7",
            population_size=6,
            generations=2,
            accuracy_fn=lambda a: 0.5 + 0.5 * min(a.total_params / 500, 1.0),
        )
        assert all(a.fitness_accuracy >= 0.5 for a in result.pareto_front)

    def test_empty_pareto_front_methods(self) -> None:
        r = NASResult(pareto_front=[], all_evaluated=[], generations=0, total_evaluations=0)
        assert r.best_accuracy() is None
        assert r.best_efficiency() is None
