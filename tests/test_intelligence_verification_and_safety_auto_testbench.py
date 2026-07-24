# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestAutoTestbench from former test_intelligence_verification_and_safety.py

"""Focused suite: TestAutoTestbench from former test_intelligence_verification_and_safety.py."""

from __future__ import annotations

from tests.intelligence_verification_and_safety_support import *  # noqa: F403


class TestAutoTestbench:
    def test_cocotb(self):
        from sc_neurocore.compiler.intelligence import generate_testbench

        tb = generate_testbench("sc_lif", {"v": "a"})
        assert "import cocotb" in tb
        assert "test_sc_lif_reset" in tb

    def test_uvm(self):
        from sc_neurocore.compiler.intelligence import generate_testbench

        tb = generate_testbench("sc_lif", {"v": "a"}, framework="uvm")
        assert "uvm_test" in tb
