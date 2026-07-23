# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCriticalPathIntegration from former test_pipeline_stages.py

"""Focused suite: TestCriticalPathIntegration from former test_pipeline_stages.py."""

from __future__ import annotations

from tests.pipeline_stages_support import *  # noqa: F403

class TestCriticalPathIntegration:
    """Verify static analysis functions work with pipeline insertion."""

    def test_lif_critical_path_depth(self):
        """LIF has 2 multiplies: reciprocal * v and reciprocal * I."""
        depth = critical_path_depth("-(v - E_L)/tau_m + I/C")
        assert depth >= 1, f"Expected depth >= 1, got {depth}"

    def test_izhikevich_critical_path_depth(self):
        """Izhikevich v-equation has 3+ chained multiplies."""
        depth = critical_path_depth("0.04 * v * v + 5 * v + 140 - u + I")
        assert depth >= 2, f"Expected depth >= 2 for v*v chain, got {depth}"

    def test_pipeline_stages_needed_low_freq(self):
        """At 100 MHz, most neurons need 0 stages."""
        stages = pipeline_stages_needed(2, 100)  # 2 DSPs at 100 MHz
        assert stages == 0

    def test_pipeline_stages_needed_high_freq(self):
        """At 900 MHz, 3+ DSPs in series need pipeline stages."""
        stages = pipeline_stages_needed(4, 900)  # 4 DSPs at 900 MHz
        assert stages >= 1, f"Expected >=1 stage at 900 MHz, got {stages}"

    def test_pipeline_analysis_full(self, izhikevich_neuron):
        """Full pipeline analysis should return per-variable results."""
        result = pipeline_analysis(izhikevich_neuron.equations, target_freq_mhz=500)
        assert "v" in result
        assert "u" in result
        assert "depth" in result["v"]
        assert "stages" in result["v"]
        assert "achievable_mhz" in result["v"]

    def test_auto_pipeline_zero_at_100mhz(self, lif_neuron):
        """Auto-pipeline at 100 MHz should give 0 stages for LIF."""
        max_depth = max(critical_path_depth(expr) for expr in lif_neuron.equations.values())
        stages = pipeline_stages_needed(max_depth, 100)
        assert stages == 0
