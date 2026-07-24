# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCompilationSummary from former test_intelligence_reporting.py

"""Focused suite: TestCompilationSummary from former test_intelligence_reporting.py."""

from __future__ import annotations

from tests.intelligence_reporting_support import *  # noqa: F403


class TestCompilationSummary:
    """End-to-end compilation summary generation."""

    def test_summary_contains_sections(self):
        from sc_neurocore.compiler.intelligence import (
            generate_compilation_summary,
        )

        s = generate_compilation_summary(
            "sc_lif",
            {"v": "a * b + c"},
            "artix7",
        )
        assert "## Module:" in s
        assert "### Equations" in s
        assert "### Target Platform" in s
        assert "### Fixed-Point Configuration" in s
        assert "### Resource Estimation" in s
        assert "### Pipeline Analysis" in s
        assert "### Applicable Features" in s

    def test_fpga_features(self):
        from sc_neurocore.compiler.intelligence import (
            generate_compilation_summary,
        )

        s = generate_compilation_summary(
            "sc_lif",
            {"v": "a + b"},
            "artix7",
        )
        assert "TMR wrapper" in s
        assert "Bitstream encryption" in s

    def test_photonic_features(self):
        from sc_neurocore.compiler.intelligence import (
            generate_compilation_summary,
        )

        s = generate_compilation_summary(
            "sc_lif",
            {"v": "a + b"},
            "lightmatter_passage",
        )
        assert "MZI weight encoding" in s

    def test_neuromorphic_features(self):
        from sc_neurocore.compiler.intelligence import (
            generate_compilation_summary,
        )

        s = generate_compilation_summary(
            "sc_lif",
            {"v": "a + b"},
            "loihi2",
        )
        assert "On-chip learning" in s

    def test_in_memory_features(self):
        from sc_neurocore.compiler.intelligence import (
            generate_compilation_summary,
        )

        s = generate_compilation_summary(
            "sc_lif",
            {"v": "a + b"},
            "upmem_pim",
        )
        assert "PIM layout planner" in s

    def test_verilog_lines_shown(self):
        from sc_neurocore.compiler.intelligence import (
            generate_compilation_summary,
        )

        s = generate_compilation_summary(
            "sc_lif",
            {"v": "a + b"},
            "artix7",
            verilog_lines=150,
        )
        assert "150 lines" in s
