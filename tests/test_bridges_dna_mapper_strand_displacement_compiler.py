# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestStrandDisplacementCompiler from former test_bridges_dna_mapper.py

"""Focused suite: TestStrandDisplacementCompiler from former test_bridges_dna_mapper.py."""

from __future__ import annotations

from tests.bridges_dna_mapper_support import *  # noqa: F403

class TestStrandDisplacementCompiler:
    def test_compile_and_gate(self) -> None:
        compiler = StrandDisplacementCompiler()
        gate = compiler.compile_and("input_a", "input_b", "output")
        assert isinstance(gate, DNAGate)
        assert gate.gate_type == GateType.AND

    def test_compile_or_gate(self) -> None:
        compiler = StrandDisplacementCompiler()
        gate = compiler.compile_or("a", "b", "out")
        assert gate.gate_type == GateType.OR

    def test_compile_not_gate(self) -> None:
        compiler = StrandDisplacementCompiler()
        gate = compiler.compile_not("in", "out")
        assert gate.gate_type == GateType.NOT

    def test_compile_threshold(self) -> None:
        compiler = StrandDisplacementCompiler()
        gate = compiler.compile_threshold("in", "out", threshold=2.0)
        assert gate.gate_type == GateType.THRESHOLD

    def test_compile_mux(self) -> None:
        compiler = StrandDisplacementCompiler()
        gate = compiler.compile_mux("sel", "a", "b", "out")
        assert isinstance(gate, DNAGate)

    def test_compile_amplifier(self) -> None:
        compiler = StrandDisplacementCompiler()
        gate = compiler.compile_amplifier("in", "out")
        assert isinstance(gate, DNAGate)

    def test_compile_buffer(self) -> None:
        compiler = StrandDisplacementCompiler()
        gate = compiler.compile_buffer("in", "out")
        assert isinstance(gate, DNAGate)
