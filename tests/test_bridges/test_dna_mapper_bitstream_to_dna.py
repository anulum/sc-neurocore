# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestBitstreamToDNA from former test_dna_mapper.py

"""Focused suite: TestBitstreamToDNA from former test_dna_mapper.py."""

from __future__ import annotations

from dna_mapper_support import *  # noqa: F403

class TestBitstreamToDNA:
    """High-level circuit compilation."""

    def test_simple_and_compiles(self, simple_and_circuit: DNACircuitDesign) -> None:
        assert simple_and_circuit.total_gates == 1
        assert len(simple_and_circuit.input_strands) == 2
        assert len(simple_and_circuit.output_strands) == 1

    def test_nand_two_gates(self, nand_circuit: DNACircuitDesign) -> None:
        assert nand_circuit.total_gates == 2
        types = [g.gate_type for g in nand_circuit.gates]
        assert GateType.AND in types
        assert GateType.NOT in types

    def test_total_strands_positive(self, simple_and_circuit: DNACircuitDesign) -> None:
        assert simple_and_circuit.total_strands > 0

    def test_total_nucleotides_positive(self, simple_and_circuit: DNACircuitDesign) -> None:
        assert simple_and_circuit.total_nucleotides > 0

    def test_circuit_validation(self, simple_and_circuit: DNACircuitDesign) -> None:
        warnings = simple_and_circuit.validate()
        # Warnings are acceptable; critical failures would raise
        assert isinstance(warnings, list)

    def test_design_validation_flags_gc_and_homopolymer_violations(self) -> None:
        design = DNACircuitDesign(
            input_strands=[
                DNAStrand(name="at_rich", sequence="ATATATAT", role="signal"),
                DNAStrand(name="poly_a", sequence="AAAACGTA", role="signal"),
            ]
        )

        warnings = design.validate()

        assert any("GC content" in warning for warning in warnings)
        assert any("homopolymer run" in warning for warning in warnings)

    def test_reproducible_compilation(self) -> None:
        c1 = BitstreamToDNA(seed=42)
        c2 = BitstreamToDNA(seed=42)
        gates = [{"type": "AND", "inputs": ["A", "B"], "output": "C"}]
        d1 = c1.compile_network(gates, ["A", "B"], ["C"])
        d2 = c2.compile_network(gates, ["A", "B"], ["C"])
        assert d1.input_strands[0].sequence == d2.input_strands[0].sequence

    def test_enzymatic_method(self) -> None:
        c = BitstreamToDNA(method="enzymatic", seed=42)
        design = c.compile_network(
            gates=[{"type": "NAND", "inputs": ["A", "B"], "output": "C"}],
            input_names=["A", "B"],
            output_names=["C"],
        )
        assert design.method == CompilationMethod.ENZYMATIC
        assert design.total_gates == 1

    def test_unsupported_gate_raises(self) -> None:
        c = BitstreamToDNA(method="displacement", seed=42)
        with pytest.raises(ValueError, match="Unsupported"):
            c.compile_network(
                gates=[{"type": "FOOBAR", "inputs": ["A"], "output": "B"}],
                input_names=["A"],
                output_names=["B"],
            )

    def test_multi_gate_cascade(self) -> None:
        c = BitstreamToDNA(seed=42)
        design = c.compile_network(
            gates=[
                {"type": "AND", "inputs": ["A", "B"], "output": "X"},
                {"type": "OR", "inputs": ["X", "C"], "output": "Y"},
                {"type": "NOT", "inputs": ["Y"], "output": "Z"},
            ],
            input_names=["A", "B", "C"],
            output_names=["Z"],
        )
        assert design.total_gates == 3
        assert len(design.input_strands) == 3
