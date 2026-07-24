# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestToggleCoverage from former test_uvm_gen.py

"""Focused suite: TestToggleCoverage from former test_uvm_gen.py."""

from __future__ import annotations

from uvm_gen_support import *  # noqa: F403


class TestToggleCoverage:
    def test_activity_bins(self):
        gen = UVMGenerator()
        bench = gen.generate(lif_module())
        assert "activity" in bench.coverage_sv

    def test_toggle_disabled(self):
        cov = CoverageSpec(toggle_coverage=False)
        gen = UVMGenerator(coverage=cov)
        bench = gen.generate(lif_module())
        assert "activity" not in bench.coverage_sv
