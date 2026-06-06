# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for pipeline stage analysis

"""Tests for pipeline stage analysis utilities."""

from __future__ import annotations

from sc_neurocore.compiler.pipeline_analysis import (
    critical_path_depth,
    pipeline_analysis,
    pipeline_stages_needed,
)


class TestPipelineAnalysis:
    """Verify static analysis functions for pipeline insertion."""

    def test_lif_critical_path_depth(self) -> None:
        """LIF has 2 multiplies: reciprocal * v and reciprocal * I."""
        depth = critical_path_depth("-(v - E_L)/tau_m + I/C")
        assert depth >= 1, f"Expected depth >= 1, got {depth}"

    def test_izhikevich_critical_path_depth(self) -> None:
        """Izhikevich v-equation has 3+ chained multiplies."""
        depth = critical_path_depth("0.04 * v * v + 5 * v + 140 - u + I")
        assert depth >= 2, f"Expected depth >= 2 for v*v chain, got {depth}"

    def test_pipeline_stages_needed_low_freq(self) -> None:
        """At 100 MHz, most neurons need 0 stages."""
        stages = pipeline_stages_needed(2, 100)  # 2 DSPs at 100 MHz
        assert stages == 0

    def test_pipeline_stages_needed_high_freq(self) -> None:
        """At 900 MHz, 3+ DSPs in series need pipeline stages."""
        stages = pipeline_stages_needed(4, 900)  # 4 DSPs at 900 MHz
        assert stages >= 1, f"Expected >=1 stage at 900 MHz, got {stages}"

    def test_pipeline_analysis_full(self) -> None:
        """Full pipeline analysis should return per-variable results."""
        eqs = {
            "v": "0.04 * v * v + 5 * v + 140 - u + I",
            "u": "a * (b * v - u)",
        }
        result = pipeline_analysis(eqs, target_freq_mhz=500)
        assert "v" in result
        assert "u" in result
        assert "depth" in result["v"]
        assert "stages" in result["v"]
        assert "achievable_mhz" in result["v"]
