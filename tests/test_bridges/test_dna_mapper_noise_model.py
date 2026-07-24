# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestNoiseModel from former test_dna_mapper.py

"""Focused suite: TestNoiseModel from former test_dna_mapper.py."""

from __future__ import annotations

from dna_mapper_support import *  # noqa: F403


class TestNoiseModel:
    """Monte Carlo noise analysis."""

    def test_sensitivity_analysis_runs(self, simple_and_circuit: DNACircuitDesign) -> None:
        nm = NoiseModel(n_trials=10, seed=42)
        report = nm.sensitivity_analysis(
            simple_and_circuit,
            {"A": 200.0, "B": 200.0},
            duration_s=600.0,
        )
        assert "n_trials" in report
        assert report["n_trials"] == 10
        assert "outputs" in report

    def test_output_statistics(self, simple_and_circuit: DNACircuitDesign) -> None:
        nm = NoiseModel(n_trials=10, seed=42)
        report = nm.sensitivity_analysis(
            simple_and_circuit,
            {"A": 200.0, "B": 200.0},
            duration_s=600.0,
        )
        for key, stats in report["outputs"].items():
            assert "mean" in stats
            assert "std" in stats
            assert "cv" in stats
            assert "robust" in stats
            assert stats["mean"] >= 0
