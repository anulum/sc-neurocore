# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestConcentrationOptimizer from former test_dna_mapper.py

"""Focused suite: TestConcentrationOptimizer from former test_dna_mapper.py."""

from __future__ import annotations

from dna_mapper_support import *  # noqa: F403


class TestConcentrationOptimizer:
    """Concentration optimization."""

    def test_optimizer_returns_result(self, simple_and_circuit: DNACircuitDesign) -> None:
        opt = ConcentrationOptimizer(n_evaluations=5, seed=42)
        truth_table = [
            {"inputs": {"A": 200.0, "B": 200.0}, "expected": {"C": "high"}},
            {"inputs": {"A": 200.0, "B": 0.0}, "expected": {"C": "low"}},
        ]
        result = opt.optimize(simple_and_circuit, truth_table, duration_s=300.0)
        assert "best_score" in result
        assert "initial_score" in result
        assert "improvement_pct" in result
        assert result["best_score"] >= 0
