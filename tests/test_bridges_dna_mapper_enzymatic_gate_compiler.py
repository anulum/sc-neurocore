# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestEnzymaticGateCompiler from former test_bridges_dna_mapper.py

"""Focused suite: TestEnzymaticGateCompiler from former test_bridges_dna_mapper.py."""

from __future__ import annotations

from tests.bridges_dna_mapper_support import *  # noqa: F403


class TestEnzymaticGateCompiler:
    def test_compile_nand(self) -> None:
        compiler = EnzymaticGateCompiler()
        gate = compiler.compile_nand("a", "b", "out")
        assert isinstance(gate, DNAGate)

    def test_compile_xor(self) -> None:
        compiler = EnzymaticGateCompiler()
        gate = compiler.compile_xor("a", "b", "out")
        assert isinstance(gate, DNAGate)
