# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestDegradationModel from former test_dna_mapper.py

"""Focused suite: TestDegradationModel from former test_dna_mapper.py."""

from __future__ import annotations

from dna_mapper_support import *  # noqa: F403


class TestDegradationModel:
    """Time-dependent DNA degradation."""

    def test_concentration_decreases(self) -> None:
        dm = DegradationModel(half_life_hr=24.0)
        remaining = dm.predict_concentration(200.0, 30, 24.0)
        assert remaining < 200.0

    def test_zero_time_no_degradation(self) -> None:
        dm = DegradationModel()
        remaining = dm.predict_concentration(200.0, 30, 0.0)
        assert abs(remaining - 200.0) < 1e-6

    def test_design_analysis(self, simple_and_circuit: DNACircuitDesign) -> None:
        dm = DegradationModel()
        report = dm.analyze_design(simple_and_circuit, time_hr=4.0)
        assert "min_remaining_pct" in report
        assert "strands" in report
        assert len(report["strands"]) > 0
        for s in report["strands"]:
            assert s["pct_remaining"] <= 100.0
