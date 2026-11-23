# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCoverageTarget from former test_uvm_gen.py

"""Focused suite: TestCoverageTarget from former test_uvm_gen.py."""

from __future__ import annotations

from sc_neurocore.uvm_gen.uvm_gen import (
    CoverageSpec,
    UVMGenerator,
)
from uvm_gen_support import (
    lif_module,
)


class TestCoverageTarget:
    """The coverage target of the generated testbench."""

    def test_target_percent_in_coverage(self) -> None:
        """The default target of 95 percent reaches the coverage file."""
        gen = UVMGenerator()
        bench = gen.generate(lif_module())
        assert "coverage_target" in bench.coverage_sv
        assert "95.0" in bench.coverage_sv

    def test_custom_target(self) -> None:
        """A custom target of 99 percent reaches the coverage file."""
        cov = CoverageSpec(target_percent=99.0)
        gen = UVMGenerator(coverage=cov)
        bench = gen.generate(lif_module())
        assert "99.0" in bench.coverage_sv

    def test_warning_on_miss(self) -> None:
        """Missing the target raises a warning."""
        gen = UVMGenerator()
        bench = gen.generate(lif_module())
        assert "uvm_warning" in bench.coverage_sv.lower() or "warning" in bench.coverage_sv.lower()
