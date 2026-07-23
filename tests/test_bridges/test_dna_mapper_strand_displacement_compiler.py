# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestStrandDisplacementCompiler from former test_dna_mapper.py

"""Focused suite: TestStrandDisplacementCompiler from former test_dna_mapper.py."""

from __future__ import annotations

from dna_mapper_support import *  # noqa: F403

class TestStrandDisplacementCompiler:
    """Gate compilation correctness."""

    def test_and_gate_structure(self, displacement_compiler: StrandDisplacementCompiler) -> None:
        gate = displacement_compiler.compile_and("A", "B", "C")
        assert gate.gate_type == GateType.AND
        assert gate.input_names == ["A", "B"]
        assert gate.output_name == "C"
        assert gate.strand_count >= 4

    def test_or_gate_structure(self, displacement_compiler: StrandDisplacementCompiler) -> None:
        gate = displacement_compiler.compile_or("A", "B", "C")
        assert gate.gate_type == GateType.OR
        assert gate.strand_count >= 3

    def test_not_gate_structure(self, displacement_compiler: StrandDisplacementCompiler) -> None:
        gate = displacement_compiler.compile_not("A", "B")
        assert gate.gate_type == GateType.NOT
        assert len(gate.input_names) == 1

    def test_threshold_gate(self, displacement_compiler: StrandDisplacementCompiler) -> None:
        gate = displacement_compiler.compile_threshold("A", "B", 0.7)
        assert gate.gate_type == GateType.THRESHOLD
        assert gate.threshold == 0.7

    def test_gate_ids_increment(self, displacement_compiler: StrandDisplacementCompiler) -> None:
        g1 = displacement_compiler.compile_and("A", "B", "C")
        g2 = displacement_compiler.compile_or("D", "E", "F")
        assert g2.gate_id == g1.gate_id + 1

    def test_leak_rate_positive(self, displacement_compiler: StrandDisplacementCompiler) -> None:
        gate = displacement_compiler.compile_and("A", "B", "C")
        assert gate.leak_rate > 0
        assert gate.leak_rate < 1e-4

    def test_leak_rate_depends_on_blocker_complementarity(
        self, displacement_compiler: StrandDisplacementCompiler
    ) -> None:
        strand = DNAStrand(name="strand", sequence="ACGTTGCAACGTTGCA")
        matched = DNAStrand(name="matched", sequence=strand.complement)
        unrelated = DNAStrand(name="unrelated", sequence="AAAAAAAAAAAAAAAA")

        matched_leak = displacement_compiler._estimate_leak_rate(strand, matched)
        unrelated_leak = displacement_compiler._estimate_leak_rate(strand, unrelated)

        assert matched_leak < unrelated_leak
