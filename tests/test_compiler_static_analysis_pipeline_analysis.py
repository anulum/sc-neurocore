# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPipelineAnalysis from former test_compiler_static_analysis.py

"""Focused suite: TestPipelineAnalysis from former test_compiler_static_analysis.py."""

from __future__ import annotations

from tests.compiler_static_analysis_support import *  # noqa: F403

class TestPipelineAnalysis:
    """Tests for critical path depth and pipeline budget."""

    def test_no_multiply(self) -> None:
        """Pure addition has zero depth."""
        from sc_neurocore.compiler.static_analysis import critical_path_depth

        assert critical_path_depth("a + b + c") == 0

    def test_single_multiply(self) -> None:
        """Single multiply has depth 1."""
        from sc_neurocore.compiler.static_analysis import critical_path_depth

        assert critical_path_depth("a * b") == 1

    def test_chained_multiply(self) -> None:
        """Chained a * b * c has depth 2."""
        from sc_neurocore.compiler.static_analysis import critical_path_depth

        assert critical_path_depth("a * b * c") == 2

    def test_deep_chain(self) -> None:
        """a * b * c * d has depth 3."""
        from sc_neurocore.compiler.static_analysis import critical_path_depth

        assert critical_path_depth("a * b * c * d") == 3

    def test_mixed(self) -> None:
        """a * b + c * d: both branches have depth 1."""
        from sc_neurocore.compiler.static_analysis import critical_path_depth

        assert critical_path_depth("a * b + c * d") == 1

    def test_divide_counts(self) -> None:
        """Division counts as multiplicative depth."""
        from sc_neurocore.compiler.static_analysis import critical_path_depth

        assert critical_path_depth("a / b") == 1

    def test_no_pipeline_needed_slow(self) -> None:
        """No pipeline at 100 MHz with depth 1."""
        from sc_neurocore.compiler.static_analysis import pipeline_stages_needed

        assert pipeline_stages_needed(1, 100) == 0

    def test_pipeline_needed_fast(self) -> None:
        """Pipeline needed at 900 MHz with depth 4."""
        from sc_neurocore.compiler.static_analysis import pipeline_stages_needed

        stages = pipeline_stages_needed(4, 900)
        assert stages >= 1  # 4 × 3.0 ns = 12 ns > 1.11 ns period

    def test_pipeline_zero_depth(self) -> None:
        """Zero depth → zero stages."""
        from sc_neurocore.compiler.static_analysis import pipeline_stages_needed

        assert pipeline_stages_needed(0, 900) == 0

    def test_pipeline_analysis_multi(self) -> None:
        """Multi-ODE pipeline analysis."""
        from sc_neurocore.compiler.static_analysis import pipeline_analysis

        result = pipeline_analysis(
            {"v": "a * b * c + d", "w": "e + f"},
            target_freq_mhz=500,
        )
        assert result["v"]["depth"] == 2
        assert result["w"]["depth"] == 0
        assert result["w"]["stages"] == 0
        assert "achievable_mhz" in result["v"]
