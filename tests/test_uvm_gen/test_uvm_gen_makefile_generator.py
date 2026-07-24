# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestMakefileGenerator from former test_uvm_gen.py

"""Focused suite: TestMakefileGenerator from former test_uvm_gen.py."""

from __future__ import annotations

from uvm_gen_support import *  # noqa: F403


class TestMakefileGenerator:
    def test_makefile_generated(self):
        gen = UVMGenerator()
        bench = gen.generate(lif_module())
        assert len(bench.makefile) > 0

    def test_makefile_has_targets(self):
        gen = UVMGenerator()
        bench = gen.generate(lif_module())
        assert "compile:" in bench.makefile
        assert "sim:" in bench.makefile
        assert "coverage:" in bench.makefile
        assert "clean:" in bench.makefile

    def test_makefile_has_regression(self):
        gen = UVMGenerator()
        bench = gen.generate(lif_module())
        assert "regression:" in bench.makefile

    def test_sim_targets_defined(self):
        assert "vcs" in SIM_TARGETS
        assert "questa" in SIM_TARGETS
        assert "xcelium" in SIM_TARGETS
