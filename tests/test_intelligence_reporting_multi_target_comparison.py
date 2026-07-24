# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestMultiTargetComparison from former test_intelligence_reporting.py

"""Focused suite: TestMultiTargetComparison from former test_intelligence_reporting.py."""

from __future__ import annotations

from tests.intelligence_reporting_support import *  # noqa: F403


class TestMultiTargetComparison:
    """Compile-once, compare-N-targets."""

    def test_compare_three_targets(self):
        from sc_neurocore.compiler.intelligence import compare_targets

        results = compare_targets(
            {"v": "a * b + c"},
            ["artix7", "ice40", "loihi2"],
        )
        assert len(results) == 3
        assert results[0].target == "artix7"
        assert results[1].target == "ice40"

    def test_dsp_targets_have_dsps(self):
        from sc_neurocore.compiler.intelligence import compare_targets

        results = compare_targets(
            {"v": "a * b"},
            ["artix7", "bae_rad750"],
        )
        artix = results[0]
        rad = results[1]
        assert artix.estimated_dsps > 0
        assert rad.estimated_dsps == 0

    def test_format_report(self):
        from sc_neurocore.compiler.intelligence import (
            compare_targets,
            format_comparison_report,
        )

        results = compare_targets(
            {"v": "a * b + c"},
            ["artix7", "loihi2"],
        )
        report = format_comparison_report(results)
        assert "Multi-Target" in report
        assert "artix7" in report
        assert "loihi2" in report
        assert "Pipeline" in report

    def test_critical_path_consistent(self):
        from sc_neurocore.compiler.intelligence import compare_targets

        results = compare_targets(
            {"v": "a * b * c"},
            ["artix7", "ice40"],
        )
        # Same equations → same depth
        assert results[0].critical_path_depth == results[1].critical_path_depth
