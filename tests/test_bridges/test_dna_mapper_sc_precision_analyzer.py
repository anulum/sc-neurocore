# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSCPrecisionAnalyzer from former test_dna_mapper.py

"""Focused suite: TestSCPrecisionAnalyzer from former test_dna_mapper.py."""

from __future__ import annotations

from dna_mapper_support import *  # noqa: F403

class TestSCPrecisionAnalyzer:
    """Stochastic computing precision analysis."""

    def test_precision_fields(self, simple_and_circuit: DNACircuitDesign) -> None:
        analyzer = SCPrecisionAnalyzer()
        result = analyzer.analyze(simple_and_circuit, {"A": 200.0, "B": 200.0})
        assert "total_effective_bits" in result
        assert result["total_effective_bits"] > 0
        assert "outputs" in result

    def test_output_statistics(self, simple_and_circuit: DNACircuitDesign) -> None:
        analyzer = SCPrecisionAnalyzer()
        result = analyzer.analyze(simple_and_circuit, {"A": 200.0, "B": 200.0})
        for key, stats in result["outputs"].items():
            assert "snr_db" in stats
            assert "effective_bits" in stats
            assert "resolution_nM" in stats
            assert "dynamic_range_db" in stats

    def test_empty_design_reports_zero_effective_bits(self) -> None:
        analyzer = SCPrecisionAnalyzer()
        result = analyzer.analyze(DNACircuitDesign(name="empty"), {})

        assert result["outputs"] == {}
        assert result["total_effective_bits"] == 0.0
