# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestEnzymaticGateCompiler from former test_dna_mapper.py

"""Focused suite: TestEnzymaticGateCompiler from former test_dna_mapper.py."""

from __future__ import annotations

from dna_mapper_support import *  # noqa: F403


class TestEnzymaticGateCompiler:
    """Enzymatic gate compilation."""

    def test_nand_gate(self, enzymatic_compiler: EnzymaticGateCompiler) -> None:
        gate = enzymatic_compiler.compile_nand("A", "B", "C")
        assert gate.gate_type == GateType.NAND
        # Check enzyme sites in substrate
        substrate = gate.strands[0].sequence
        assert "GAATTC" in substrate  # EcoRI
        assert "GGATCC" in substrate  # BamHI

    def test_xor_gate(self, enzymatic_compiler: EnzymaticGateCompiler) -> None:
        gate = enzymatic_compiler.compile_xor("A", "B", "C")
        assert gate.gate_type == GateType.XOR
        assert gate.strand_count >= 3
