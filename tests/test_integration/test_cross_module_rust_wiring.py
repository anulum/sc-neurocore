# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestRustWiring from former test_cross_module.py

"""Focused suite: TestRustWiring from former test_cross_module.py."""

from __future__ import annotations

from cross_module_support import *  # noqa: F403

class TestRustWiring:
    """Verify Rust backends are importable and callable from Python."""

    def test_optimizer_rust_import(self):
        try:
            from sc_neurocore_engine import py_opt_sa_search, py_opt_extract_pareto
        except ImportError:
            pytest.skip("Rust engine not built")
        assert callable(py_opt_sa_search)
        assert callable(py_opt_extract_pareto)

    def test_optimizer_sa_via_python_api(self):
        from sc_neurocore.optimizer.sc_optimizer import SCOptimizer, HardwareBudget, LayerProfile

        budget = HardwareBudget(max_luts=100_000, max_power_mw=1000.0)
        opt = SCOptimizer(budget)
        network = [
            LayerProfile(id="L0", mac_count=10),
            LayerProfile(id="L1", mac_count=20, is_critical_path=True),
        ]
        report = opt.optimize_annealing(network, max_iter=100)
        assert report is not None
        assert report.mean_accuracy > 0.5

    def test_evo_rust_functions(self):
        try:
            from sc_neurocore_engine import (
                py_evo_batch_mutate,
                py_evo_batch_fitness,
                py_evo_batch_crossover,
                py_evo_diversity,
                py_evo_novelty,
                py_evo_tournament,
            )
        except ImportError:
            pytest.skip("Rust engine not built")
        for _fn in (
            py_evo_batch_fitness,
            py_evo_batch_crossover,
            py_evo_diversity,
            py_evo_novelty,
            py_evo_tournament,
        ):
            assert callable(_fn)
        pop = [[0.0] * 10 for _ in range(20)]
        mutated = py_evo_batch_mutate(pop, 0.5, 0.1, 42)
        assert len(mutated) == 20
        assert any(w != 0.0 for g in mutated for w in g)

    def test_pareto_extraction_rust(self):
        try:
            from sc_neurocore_engine import py_opt_extract_pareto
        except ImportError:
            pytest.skip("Rust engine not built")
        result = py_opt_extract_pareto(
            [100, 200, 150],
            [1.0, 0.5, 0.8],
            [0.9, 0.95, 0.85],
        )
        assert "indices" in result
        assert len(result["luts"]) > 0
