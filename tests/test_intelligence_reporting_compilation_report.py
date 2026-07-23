# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCompilationReport from former test_intelligence_reporting.py

"""Focused suite: TestCompilationReport from former test_intelligence_reporting.py."""

from __future__ import annotations

from tests.intelligence_reporting_support import *  # noqa: F403

class TestCompilationReport:
    def test_basic(self):
        from sc_neurocore.compiler.intelligence import generate_compilation_report

        md = generate_compilation_report("sc_lif", {"v": "a"}, "artix7")
        assert "# SC-NeuroCore Compilation Report" in md
        assert "artix7" in md
        assert "Carbon" in md

    def test_no_carbon(self):
        from sc_neurocore.compiler.intelligence import generate_compilation_report

        md = generate_compilation_report(
            "sc_lif",
            {"v": "a"},
            "artix7",
            include_carbon=False,
        )
        assert "Carbon" not in md
