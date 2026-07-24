# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSCCCoverage from former test_uvm_gen.py

"""Focused suite: TestSCCCoverage from former test_uvm_gen.py."""

from __future__ import annotations

from uvm_gen_support import *  # noqa: F403


class TestSCCCoverage:
    def test_scc_bins_generated(self):
        gen = UVMGenerator()
        bench = gen.generate(lif_module())
        assert "scc" in bench.coverage_sv.lower()

    def test_scc_bins_count(self):
        cov = CoverageSpec(scc_bins=12)
        gen = UVMGenerator(coverage=cov)
        bench = gen.generate(lif_module())
        assert "12" in bench.coverage_sv
