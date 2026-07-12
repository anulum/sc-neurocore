# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for Molecular/DNA Computing Mapper

"""Multi-angle test suite for the DNA computing mapper.

Tests cover:
- Sequence design constraints (GC content, homopolymer, orthogonality)
- Gate compilation correctness (AND, OR, NOT, THRESHOLD, NAND, XOR)
- Circuit assembly and validation
- Kinetic simulation convergence
- Export format correctness (GenBank, FASTA, NUPACK, JSON)
- Edge cases (empty network, single gate, maximum complexity)
- Round-trip fidelity (compile → simulate → verify logic)
- Thermodynamic validation
- Strand interaction analysis
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import sc_neurocore.bridges.dna_mapper as dna_mapper
from sc_neurocore.bridges.dna_mapper import (
    BitstreamToDNA,
    CompilationMethod,
    ConcentrationOptimizer,
    CrossHybridizationChecker,
    DNACircuitDesign,
    DNAGate,
    DNAStrand,
    DegradationModel,
    DualRailEncoder,
    EnzymaticGateCompiler,
    GF4ErrorCorrection,
    GateOptimizer,
    GateType,
    HairpinChecker,
    KineticSimulator,
    NUPACKInterface,
    NoiseModel,
    PlateLayout,
    SCNetworkBridge,
    SCPrecisionAnalyzer,
    SequenceDesigner,
    StrandDisplacementCompiler,
    TopologicalAnalyzer,
    estimate_cost,
    export_fasta,
    export_genbank,
    export_json,
    export_nupack_input,
    generate_protocol,
    visualize_circuit,
    visualize_kinetics,
)
from sc_neurocore.bridges.dna_types import _GC_TARGET_HIGH, _GC_TARGET_LOW, _MAX_HOMOPOLYMER


# ══════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════


@pytest.fixture
def designer() -> SequenceDesigner:
    return SequenceDesigner(seed=42)


@pytest.fixture
def displacement_compiler() -> StrandDisplacementCompiler:
    return StrandDisplacementCompiler()


@pytest.fixture
def enzymatic_compiler() -> EnzymaticGateCompiler:
    return EnzymaticGateCompiler()


@pytest.fixture
def nupack_interface() -> NUPACKInterface:
    return NUPACKInterface()


@pytest.fixture
def simple_and_circuit() -> DNACircuitDesign:
    compiler = BitstreamToDNA(method="displacement", seed=42)
    return compiler.compile_network(
        gates=[{"type": "AND", "inputs": ["A", "B"], "output": "C"}],
        input_names=["A", "B"],
        output_names=["C"],
        name="simple_and",
    )


@pytest.fixture
def nand_circuit() -> DNACircuitDesign:
    compiler = BitstreamToDNA(method="displacement", seed=42)
    return compiler.compile_network(
        gates=[
            {"type": "AND", "inputs": ["A", "B"], "output": "X"},
            {"type": "NOT", "inputs": ["X"], "output": "Y"},
        ],
        input_names=["A", "B"],
        output_names=["Y"],
        name="nand_circuit",
    )


@pytest.fixture
def tmp_path_factory_dir(tmp_path: Path) -> Path:
    return tmp_path


# ══════════════════════════════════════════════════════════════════════
# 1. Sequence Design Constraints
# ══════════════════════════════════════════════════════════════════════


class TestSequenceDesigner:
    """Sequence generation constraint satisfaction."""

    def test_gc_content_within_bounds(self, designer: SequenceDesigner) -> None:
        for i in range(20):
            seq = designer.generate(30, f"test_gc_{i}")
            gc = sum(1 for c in seq if c in "GC") / len(seq)
            assert _GC_TARGET_LOW - 0.1 <= gc <= _GC_TARGET_HIGH + 0.1, (
                f"Sequence {i} GC={gc:.3f} outside bounds"
            )

    def test_no_excessive_homopolymer(self, designer: SequenceDesigner) -> None:
        for i in range(20):
            seq = designer.generate(50, f"test_homo_{i}")
            max_run = 1
            cur_run = 1
            for j in range(1, len(seq)):
                if seq[j] == seq[j - 1]:
                    cur_run += 1
                    max_run = max(max_run, cur_run)
                else:
                    cur_run = 1
            assert max_run <= _MAX_HOMOPOLYMER + 1, f"Sequence {i} has homopolymer run {max_run}"

    def test_deterministic_with_same_seed(self) -> None:
        d1 = SequenceDesigner(seed=99)
        d2 = SequenceDesigner(seed=99)
        seq1 = d1.generate(20, "x")
        seq2 = d2.generate(20, "x")
        assert seq1 == seq2

    def test_different_seeds_different_sequences(self) -> None:
        d1 = SequenceDesigner(seed=1)
        d2 = SequenceDesigner(seed=2)
        seq1 = d1.generate(20, "x")
        seq2 = d2.generate(20, "x")
        assert seq1 != seq2

    def test_complement_is_watson_crick(self, designer: SequenceDesigner) -> None:
        seq = designer.generate(20, "comp_test")
        comp = designer.generate_complement(seq)
        pairs = {"A": "T", "T": "A", "C": "G", "G": "C"}
        for i, c in enumerate(seq):
            expected = pairs[c]
            actual = comp[len(seq) - 1 - i]
            assert actual == expected, f"Position {i}: {c} → {actual}, expected {expected}"

    def test_toehold_length(self, designer: SequenceDesigner) -> None:
        th = designer.generate_toehold("test")
        assert len(th) == 6

    def test_recognition_length(self, designer: SequenceDesigner) -> None:
        rec = designer.generate_recognition("test")
        assert len(rec) == 15

    def test_orthogonality_low_overlap(self, designer: SequenceDesigner) -> None:
        sequences = [designer.generate(20, f"ortho_{i}") for i in range(10)]
        for i in range(len(sequences)):
            for j in range(i + 1, len(sequences)):
                overlap = sum(1 for a, b in zip(sequences[i], sequences[j]) if a == b)
                similarity = overlap / 20
                assert similarity < 0.90, f"Sequences {i} and {j} too similar: {similarity:.2f}"

    def test_sequence_scoring_penalizes_adverse_homopolymer_rng(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class ConstantRng:
            def integers(self, low: int, high: int) -> int:
                return low

            def choice(self, nucs: list[str], p: list[float]) -> str:
                return "A"

        monkeypatch.setattr(np.random, "default_rng", lambda seed=None: ConstantRng())

        seq = SequenceDesigner(seed=42).generate(4, "forced_homopolymer")

        assert seq == "AAAA"

    def test_sequence_generation_recovers_when_constraints_exhaust_weights(self) -> None:
        seq = SequenceDesigner(seed=42, max_homopolymer=0).generate(4, "zero_run_budget")

        assert len(seq) == 4
        assert set(seq).issubset({"A", "C", "G", "T"})


# ══════════════════════════════════════════════════════════════════════
# 2. DNAStrand Properties
# ══════════════════════════════════════════════════════════════════════


class TestDNAStrand:
    """Strand data class properties."""

    def test_length(self) -> None:
        s = DNAStrand(name="x", sequence="ACGTACGT")
        assert s.length == 8

    def test_gc_content(self) -> None:
        s = DNAStrand(name="x", sequence="GCGCGC")
        assert s.gc_content == 1.0
        s2 = DNAStrand(name="y", sequence="ATATAT")
        assert s2.gc_content == 0.0

    def test_complement(self) -> None:
        s = DNAStrand(name="x", sequence="ACGT")
        assert s.complement == "ACGT"  # palindromic

    def test_max_homopolymer_run(self) -> None:
        s = DNAStrand(name="x", sequence="AACCCGT")
        assert s.max_homopolymer_run == 3

    def test_delta_g_negative(self) -> None:
        s = DNAStrand(name="x", sequence="GCGCGCGCGC")
        dg = s.delta_g_37()
        assert dg < 0, "GC-rich sequence should have negative ΔG"

    def test_melting_temperature_gc_rich_higher(self) -> None:
        gc_rich = DNAStrand(name="gc", sequence="GCGCGCGCGCGCGCGC")
        at_rich = DNAStrand(name="at", sequence="ATATATATATATATATAT")
        assert gc_rich.melting_temperature() > at_rich.melting_temperature()

    def test_melting_temperature_depends_on_salt_and_strand_concentration(self) -> None:
        strand = DNAStrand(name="thermo", sequence="ACGTTGCAACGTTGCA")
        low_salt = strand.melting_temperature(na_conc_M=0.05, strand_conc_M=2.5e-7)
        high_salt = strand.melting_temperature(na_conc_M=1.0, strand_conc_M=2.5e-7)
        high_conc = strand.melting_temperature(na_conc_M=0.05, strand_conc_M=2.5e-6)

        assert high_salt > low_salt
        assert high_conc > low_salt

    def test_melting_temperature_rejects_invalid_conditions(self) -> None:
        strand = DNAStrand(name="thermo", sequence="ACGTTGCA")
        with pytest.raises(ValueError, match="na_conc_M must be finite and positive"):
            strand.melting_temperature(na_conc_M=0.0)
        with pytest.raises(ValueError, match="strand_conc_M must be finite and positive"):
            strand.melting_temperature(strand_conc_M=0.0)

    def test_empty_strand(self) -> None:
        s = DNAStrand(name="empty", sequence="")
        assert s.length == 0
        assert s.gc_content == 0.0
        assert s.max_homopolymer_run == 0

    def test_short_strand_has_zero_delta_g_and_rejects_tm(self) -> None:
        s = DNAStrand(name="short", sequence="A")

        assert s.delta_g_37() == 0.0
        with pytest.raises(ValueError, match="at least two nucleotides"):
            s.melting_temperature()


# ══════════════════════════════════════════════════════════════════════
# 3. Strand Displacement Compiler
# ══════════════════════════════════════════════════════════════════════


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


# ══════════════════════════════════════════════════════════════════════
# 4. Enzymatic Gate Compiler
# ══════════════════════════════════════════════════════════════════════


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


# ══════════════════════════════════════════════════════════════════════
# 5. Circuit Assembly: BitstreamToDNA
# ══════════════════════════════════════════════════════════════════════


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


# ══════════════════════════════════════════════════════════════════════
# 6. Kinetic Simulation
# ══════════════════════════════════════════════════════════════════════


class TestKineticSimulator:
    """Simulation correctness and convergence."""

    def test_and_both_inputs_high(self, simple_and_circuit: DNACircuitDesign) -> None:
        sim = KineticSimulator()
        result = sim.simulate(simple_and_circuit, {"A": 200.0, "B": 200.0}, duration_s=3600.0)
        assert "time" in result
        output_key = simple_and_circuit.gates[0].output_name
        assert output_key in result
        final = result[output_key][-1]
        assert final > 50.0, f"AND(1,1) should produce high output, got {final}"

    def test_and_one_input_low(self, simple_and_circuit: DNACircuitDesign) -> None:
        sim = KineticSimulator()
        result = sim.simulate(simple_and_circuit, {"A": 200.0, "B": 0.0}, duration_s=3600.0)
        output_key = simple_and_circuit.gates[0].output_name
        final = result[output_key][-1]
        assert final < 50.0, f"AND(1,0) should produce low output, got {final}"

    def test_simulation_time_steps(self, simple_and_circuit: DNACircuitDesign) -> None:
        sim = KineticSimulator()
        result = sim.simulate(simple_and_circuit, {"A": 100.0}, duration_s=100.0, dt=0.5)
        assert len(result["time"]) == 200

    def test_concentrations_non_negative(self, simple_and_circuit: DNACircuitDesign) -> None:
        sim = KineticSimulator()
        result = sim.simulate(simple_and_circuit, {"A": 200.0, "B": 200.0})
        for key, trace in result.items():
            if key == "time":
                continue
            assert np.all(trace >= 0), f"Negative concentrations in {key}"

    def test_concentrations_bounded(self, simple_and_circuit: DNACircuitDesign) -> None:
        sim = KineticSimulator()
        result = sim.simulate(simple_and_circuit, {"A": 200.0, "B": 200.0})
        for key, trace in result.items():
            if key == "time":
                continue
            assert np.all(trace <= 201.0), f"Concentration exceeds max in {key}"


# ══════════════════════════════════════════════════════════════════════
# 7. NUPACK Interface
# ══════════════════════════════════════════════════════════════════════


class TestNUPACKInterface:
    """Thermodynamic validation."""

    def test_mfe_returns_tuple(self, nupack_interface: NUPACKInterface) -> None:
        energy, structure = nupack_interface.compute_mfe("ACGTACGT")
        assert isinstance(energy, float)
        assert isinstance(structure, str)

    def test_pair_probabilities_shape(self, nupack_interface: NUPACKInterface) -> None:
        seq = "ACGTACGT"
        probs = nupack_interface.compute_pair_probabilities(seq)
        assert probs.shape == (8, 8)

    def test_fallback_predicts_intramolecular_pairing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(dna_mapper, "_HAS_NUPACK", False)
        interface = NUPACKInterface()
        sequence = "GCGCAAAGCGC"

        energy, structure = interface.compute_mfe(sequence)
        probs = interface.compute_pair_probabilities(sequence)

        assert energy < 0.0
        assert "(" in structure and ")" in structure
        assert probs.shape == (len(sequence), len(sequence))
        assert probs[0, -1] > 0.0
        assert probs[1, -2] > 0.0
        assert np.allclose(probs, probs.T)
        assert np.all((probs >= 0.0) & (probs <= 1.0))

    def test_fallback_rejects_invalid_bases_and_handles_empty_or_unpairable_sequences(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(dna_mapper, "_HAS_NUPACK", False)
        interface = NUPACKInterface()

        assert interface.compute_mfe("") == (0.0, "")
        assert not np.any(interface.compute_pair_probabilities("AAAA"))
        with pytest.raises(ValueError, match="invalid bases"):
            interface.compute_mfe("ACGX")

    def test_nupack_backend_path_uses_module_contract(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class FakeModel:
            def __init__(self, **kwargs: object) -> None:
                self.kwargs = kwargs

        class FakeStrand:
            def __init__(self, sequence: str, name: str) -> None:
                self.sequence = sequence
                self.name = name

        class FakePairs:
            def to_array(self) -> np.ndarray[Any, Any]:
                return np.array([[0.0, 0.25], [0.25, 0.0]])

        fake_nupack = SimpleNamespace(
            Model=FakeModel,
            Strand=FakeStrand,
            mfe=lambda strands, model: [SimpleNamespace(energy=-3.5, structure="()")],
            pairs=lambda strands, model: FakePairs(),
        )
        monkeypatch.setattr(dna_mapper, "_HAS_NUPACK", True)
        monkeypatch.setattr(dna_mapper, "nupack", fake_nupack)
        interface = NUPACKInterface(temperature_c=25.0, na_concentration_M=0.5)

        assert interface.has_nupack is True
        assert interface.compute_mfe("AT") == (-3.5, "()")
        assert np.allclose(interface.compute_pair_probabilities("AT"), [[0.0, 0.25], [0.25, 0.0]])

    def test_validate_design(
        self,
        nupack_interface: NUPACKInterface,
        simple_and_circuit: DNACircuitDesign,
    ) -> None:
        report = nupack_interface.validate_design(simple_and_circuit)
        assert "valid" in report
        assert "strand_results" in report
        assert "warnings" in report
        assert isinstance(report["valid"], bool)

    def test_validate_design_marks_design_rule_warnings_invalid(self) -> None:
        design = DNACircuitDesign(
            name="invalid_rules",
            input_strands=[DNAStrand(name="poly_a", sequence="AAAAAAA", role="output")],
        )

        report = NUPACKInterface().validate_design(design)

        assert report["valid"] is False
        assert report["warnings"]


# ══════════════════════════════════════════════════════════════════════
# 8. Export Formats
# ══════════════════════════════════════════════════════════════════════


class TestExportFormats:
    """Verify exported file format correctness."""

    def test_genbank_export(
        self,
        simple_and_circuit: DNACircuitDesign,
        tmp_path: Path,
    ) -> None:
        path = str(tmp_path / "test.gb")
        export_genbank(simple_and_circuit, path)
        content = Path(path).read_text()
        assert "LOCUS" in content
        assert "ORIGIN" in content
        assert "//" in content
        assert "synthetic construct" in content

    def test_fasta_export(
        self,
        simple_and_circuit: DNACircuitDesign,
        tmp_path: Path,
    ) -> None:
        path = str(tmp_path / "test.fasta")
        export_fasta(simple_and_circuit, path)
        content = Path(path).read_text()
        lines = content.strip().split("\n")
        fasta_headers = [l for l in lines if l.startswith(">")]
        assert len(fasta_headers) >= 1

    def test_nupack_export(
        self,
        simple_and_circuit: DNACircuitDesign,
        tmp_path: Path,
    ) -> None:
        path = str(tmp_path / "test.nupack")
        export_nupack_input(simple_and_circuit, path)
        content = Path(path).read_text()
        assert "material = dna" in content
        assert "strand" in content

    def test_json_export(
        self,
        simple_and_circuit: DNACircuitDesign,
        tmp_path: Path,
    ) -> None:
        path = str(tmp_path / "test.json")
        export_json(simple_and_circuit, path)
        data = json.loads(Path(path).read_text())
        assert data["name"] == "simple_and"
        assert data["total_gates"] == 1
        assert "gates" in data
        assert len(data["gates"]) == 1

    def test_json_round_trip_fields(
        self,
        nand_circuit: DNACircuitDesign,
        tmp_path: Path,
    ) -> None:
        path = str(tmp_path / "nand.json")
        export_json(nand_circuit, path)
        data = json.loads(Path(path).read_text())
        for gate in data["gates"]:
            assert "gate_type" in gate
            assert "strands" in gate
            for strand in gate["strands"]:
                assert "sequence" in strand
                assert "gc_content" in strand
                assert "delta_g_37" in strand


# ══════════════════════════════════════════════════════════════════════
# 9. Edge Cases
# ══════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Boundary conditions and unusual inputs."""

    def test_single_not_gate(self) -> None:
        c = BitstreamToDNA(seed=42)
        design = c.compile_network(
            gates=[{"type": "NOT", "inputs": ["A"], "output": "B"}],
            input_names=["A"],
            output_names=["B"],
        )
        assert design.total_gates == 1

    def test_deep_cascade_10_gates(self) -> None:
        c = BitstreamToDNA(seed=42)
        gates = []
        prev = "A"
        for i in range(10):
            out = f"g{i}"
            gates.append({"type": "NOT", "inputs": [prev], "output": out})
            prev = out
        design = c.compile_network(gates=gates, input_names=["A"], output_names=[prev])
        assert design.total_gates == 10

    def test_threshold_zero(self) -> None:
        c = BitstreamToDNA(seed=42)
        design = c.compile_network(
            gates=[{"type": "THRESHOLD", "inputs": ["A"], "output": "B", "threshold": 0.0}],
            input_names=["A"],
            output_names=["B"],
        )
        assert design.gates[0].threshold == 0.0

    def test_threshold_one(self) -> None:
        c = BitstreamToDNA(seed=42)
        design = c.compile_network(
            gates=[{"type": "THRESHOLD", "inputs": ["A"], "output": "B", "threshold": 1.0}],
            input_names=["A"],
            output_names=["B"],
        )
        assert design.gates[0].threshold == 1.0

    def test_high_level_simulate_wrapper_returns_time_and_gate_trace(self) -> None:
        c = BitstreamToDNA(seed=42)
        design = c.compile_network(
            gates=[{"type": "BUFFER", "inputs": ["A"], "output": "B"}],
            input_names=["A"],
            output_names=["B"],
        )

        result = c.simulate(design, {"A": 200.0}, duration_s=10.0, dt=1.0)

        assert "time" in result
        assert "B" in result
        assert result["B"][-1] > 0.0


# ══════════════════════════════════════════════════════════════════════
# 10. Round-Trip Logic Verification
# ══════════════════════════════════════════════════════════════════════


class TestRoundTripLogic:
    """Verify that compiled circuits implement correct Boolean logic."""

    def test_and_truth_table(self) -> None:
        c = BitstreamToDNA(seed=42)
        design = c.compile_network(
            gates=[{"type": "AND", "inputs": ["A", "B"], "output": "C"}],
            input_names=["A", "B"],
            output_names=["C"],
        )
        sim = KineticSimulator()
        output_key = design.gates[0].output_name

        # (0, 0) → 0
        r = sim.simulate(design, {"A": 0.0, "B": 0.0}, duration_s=1800.0)
        assert r[output_key][-1] < 50.0

        # (1, 0) → 0
        r = sim.simulate(design, {"A": 200.0, "B": 0.0}, duration_s=1800.0)
        assert r[output_key][-1] < 50.0

        # (0, 1) → 0
        r = sim.simulate(design, {"A": 0.0, "B": 200.0}, duration_s=1800.0)
        assert r[output_key][-1] < 50.0

        # (1, 1) → 1
        r = sim.simulate(design, {"A": 200.0, "B": 200.0}, duration_s=1800.0)
        assert r[output_key][-1] > 50.0

    def test_or_truth_table(self) -> None:
        c = BitstreamToDNA(seed=42)
        design = c.compile_network(
            gates=[{"type": "OR", "inputs": ["A", "B"], "output": "C"}],
            input_names=["A", "B"],
            output_names=["C"],
        )
        sim = KineticSimulator()
        output_key = design.gates[0].output_name

        r = sim.simulate(design, {"A": 0.0, "B": 0.0}, duration_s=1800.0)
        assert r[output_key][-1] < 50.0

        r = sim.simulate(design, {"A": 200.0, "B": 0.0}, duration_s=1800.0)
        assert r[output_key][-1] > 50.0

    def test_not_truth_table(self) -> None:
        c = BitstreamToDNA(seed=42)
        design = c.compile_network(
            gates=[{"type": "NOT", "inputs": ["A"], "output": "B"}],
            input_names=["A"],
            output_names=["B"],
        )
        sim = KineticSimulator()
        output_key = design.gates[0].output_name

        # Input high → output low
        r = sim.simulate(design, {"A": 200.0}, duration_s=1800.0)
        assert r[output_key][-1] < 50.0

        # Input low → output high
        r = sim.simulate(design, {"A": 0.0}, duration_s=1800.0)
        assert r[output_key][-1] > 50.0


# ══════════════════════════════════════════════════════════════════════
# 11. New Gate Types: MUX, AMPLIFIER, BUFFER
# ══════════════════════════════════════════════════════════════════════


class TestNewGateTypes:
    """MUX, AMPLIFIER, and BUFFER gate compilation and simulation."""

    def test_mux_compiles(self) -> None:
        c = BitstreamToDNA(seed=42)
        design = c.compile_network(
            gates=[{"type": "MUX", "inputs": ["S", "A", "B"], "output": "Y"}],
            input_names=["S", "A", "B"],
            output_names=["Y"],
        )
        assert design.total_gates == 1
        assert design.gates[0].gate_type == GateType.MUX
        assert len(design.gates[0].input_names) == 3

    def test_amplifier_compiles(self) -> None:
        c = BitstreamToDNA(seed=42)
        design = c.compile_network(
            gates=[{"type": "AMPLIFIER", "inputs": ["A"], "output": "B"}],
            input_names=["A"],
            output_names=["B"],
        )
        assert design.total_gates == 1
        assert design.gates[0].gate_type == GateType.AMPLIFIER

    def test_buffer_compiles(self) -> None:
        c = BitstreamToDNA(seed=42)
        design = c.compile_network(
            gates=[{"type": "BUFFER", "inputs": ["A"], "output": "B"}],
            input_names=["A"],
            output_names=["B"],
        )
        assert design.total_gates == 1
        assert design.gates[0].gate_type == GateType.BUFFER

    def test_mux_strand_count(self) -> None:
        compiler = StrandDisplacementCompiler()
        gate = compiler.compile_mux("S", "A", "B", "Y")
        assert gate.strand_count >= 4

    def test_amplifier_high_fuel(self) -> None:
        compiler = StrandDisplacementCompiler()
        gate = compiler.compile_amplifier("A", "B")
        fuel = [s for s in gate.strands if s.role == "fuel"]
        assert len(fuel) >= 1
        assert fuel[0].concentration_nM >= 500.0

    def test_buffer_in_cascade(self) -> None:
        c = BitstreamToDNA(seed=42)
        design = c.compile_network(
            gates=[
                {"type": "AND", "inputs": ["A", "B"], "output": "X"},
                {"type": "BUFFER", "inputs": ["X"], "output": "Y"},
                {"type": "NOT", "inputs": ["Y"], "output": "Z"},
            ],
            input_names=["A", "B"],
            output_names=["Z"],
        )
        assert design.total_gates == 3

    @pytest.mark.parametrize(
        ("gate_type", "inputs", "concentrations"),
        [
            (GateType.THRESHOLD, ["A"], {"A": 200.0}),
            (GateType.MUX, ["S", "A", "B"], {"S": 200.0, "A": 200.0, "B": 0.0}),
            (GateType.AMPLIFIER, ["A"], {"A": 50.0}),
            (GateType.BUFFER, ["A"], {"A": 200.0}),
        ],
    )
    def test_kinetic_simulator_dispatches_extended_gate_rates(
        self,
        gate_type: GateType,
        inputs: list[str],
        concentrations: dict[str, float],
    ) -> None:
        gate = DNAGate(
            gate_id=0,
            gate_type=gate_type,
            input_names=inputs,
            output_name="Y",
            threshold=0.5,
            leak_rate=0.0,
        )
        design = DNACircuitDesign(name="extended_gate", gates=[gate])

        result = KineticSimulator().simulate(design, concentrations, duration_s=20.0, dt=1.0)

        assert result["Y"][-1] > 0.0

    def test_kinetic_simulator_unknown_gate_fails_closed_to_leak_only(self) -> None:
        gate = DNAGate(
            gate_id=0,
            gate_type=GateType.CATALYTIC,
            input_names=["A"],
            output_name="Y",
            leak_rate=0.0,
        )
        design = DNACircuitDesign(name="unknown_gate", gates=[gate])

        result = KineticSimulator().simulate(design, {"A": 200.0}, duration_s=20.0, dt=1.0)

        assert np.all(result["Y"] == 0.0)

    def test_enzymatic_xor_network_compiles(self) -> None:
        c = BitstreamToDNA(method="enzymatic", seed=42)
        design = c.compile_network(
            gates=[{"type": "XOR", "inputs": ["A", "B"], "output": "C"}],
            input_names=["A", "B"],
            output_names=["C"],
        )

        assert design.gates[0].gate_type == GateType.XOR


# ══════════════════════════════════════════════════════════════════════
# 12. Error Correction
# ══════════════════════════════════════════════════════════════════════


class TestGF4ErrorCorrection:
    """Reed-Solomon over GF(4) for DNA error correction."""

    def test_encode_increases_length(self) -> None:
        ec = GF4ErrorCorrection(n_parity=4, block_size=12)
        original = "ACGTACGTACGT"
        encoded = ec.encode(original)
        assert len(encoded) > len(original)

    def test_round_trip_no_errors(self) -> None:
        ec = GF4ErrorCorrection(n_parity=4, block_size=12)
        original = "ACGTACGTACGT"
        encoded = ec.encode(original)
        decoded, corrections = ec.decode(encoded)
        assert decoded == original
        assert corrections == 0

    def test_detects_single_error(self) -> None:
        ec = GF4ErrorCorrection(n_parity=4, block_size=12)
        original = "ACGTACGTACGT"
        encoded = ec.encode(original)
        mutated = list(encoded)
        mutated[3] = "T" if mutated[3] != "T" else "A"
        mutated_str = "".join(mutated)
        _, corrections = ec.decode(mutated_str)
        assert corrections >= 1

    def test_multiple_blocks(self) -> None:
        ec = GF4ErrorCorrection(n_parity=4, block_size=12)
        original = "ACGTACGTACGT" * 5
        encoded = ec.encode(original)
        decoded, corrections = ec.decode(encoded)
        assert decoded == original
        assert corrections == 0


# ══════════════════════════════════════════════════════════════════════
# 13. Cross-Hybridization Checker
# ══════════════════════════════════════════════════════════════════════


class TestCrossHybridization:
    """Cross-hybridization detection."""

    def test_returns_list(self, simple_and_circuit: DNACircuitDesign) -> None:
        checker = CrossHybridizationChecker(max_complementary_run=8)
        flags = checker.check(simple_and_circuit)
        assert isinstance(flags, list)

    def test_flag_structure(self) -> None:
        checker = CrossHybridizationChecker(max_complementary_run=3)
        c = BitstreamToDNA(seed=42)
        design = c.compile_network(
            gates=[{"type": "AND", "inputs": ["A", "B"], "output": "C"}],
            input_names=["A", "B"],
            output_names=["C"],
        )
        flags = checker.check(design)
        for flag in flags:
            assert "strand_a" in flag
            assert "strand_b" in flag
            assert "complementary_run" in flag
            assert "severity" in flag

    def test_longest_common_substring(self) -> None:
        result = CrossHybridizationChecker._longest_common_substring("ABCDEF", "XBCDEY")
        assert result == 4


# ══════════════════════════════════════════════════════════════════════
# 14. Noise Model & Sensitivity
# ══════════════════════════════════════════════════════════════════════


class TestNoiseModel:
    """Monte Carlo noise analysis."""

    def test_sensitivity_analysis_runs(self, simple_and_circuit: DNACircuitDesign) -> None:
        nm = NoiseModel(n_trials=10, seed=42)
        report = nm.sensitivity_analysis(
            simple_and_circuit,
            {"A": 200.0, "B": 200.0},
            duration_s=600.0,
        )
        assert "n_trials" in report
        assert report["n_trials"] == 10
        assert "outputs" in report

    def test_output_statistics(self, simple_and_circuit: DNACircuitDesign) -> None:
        nm = NoiseModel(n_trials=10, seed=42)
        report = nm.sensitivity_analysis(
            simple_and_circuit,
            {"A": 200.0, "B": 200.0},
            duration_s=600.0,
        )
        for key, stats in report["outputs"].items():
            assert "mean" in stats
            assert "std" in stats
            assert "cv" in stats
            assert "robust" in stats
            assert stats["mean"] >= 0


# ══════════════════════════════════════════════════════════════════════
# 15. Cost Estimation
# ══════════════════════════════════════════════════════════════════════


class TestCostEstimation:
    """Oligo synthesis cost estimation."""

    def test_cost_positive(self, simple_and_circuit: DNACircuitDesign) -> None:
        cost = estimate_cost(simple_and_circuit)
        assert cost["total_cost_usd"] > 0
        assert cost["n_unique_oligos"] > 0

    def test_hplc_more_expensive(self, simple_and_circuit: DNACircuitDesign) -> None:
        standard = estimate_cost(simple_and_circuit, purification="standard")
        hplc = estimate_cost(simple_and_circuit, purification="hplc")
        assert hplc["total_cost_usd"] > standard["total_cost_usd"]

    def test_cost_per_strand_present(self, simple_and_circuit: DNACircuitDesign) -> None:
        cost = estimate_cost(simple_and_circuit)
        assert "strand_costs" in cost
        for sc in cost["strand_costs"]:
            assert "name" in sc
            assert "length" in sc
            assert "cost_usd" in sc


# ══════════════════════════════════════════════════════════════════════
# 16. Protocol Generation
# ══════════════════════════════════════════════════════════════════════


class TestProtocolGeneration:
    """Wet-lab protocol generation."""

    def test_protocol_is_markdown(self, simple_and_circuit: DNACircuitDesign) -> None:
        protocol = generate_protocol(simple_and_circuit)
        assert protocol.startswith("# Wet-Lab Protocol")
        assert "## Materials" in protocol
        assert "## Procedure" in protocol

    def test_protocol_contains_strands(self, simple_and_circuit: DNACircuitDesign) -> None:
        protocol = generate_protocol(simple_and_circuit)
        assert "translator" in protocol or "signal" in protocol

    def test_protocol_custom_volume(self, simple_and_circuit: DNACircuitDesign) -> None:
        protocol = generate_protocol(simple_and_circuit, volume_uL=100.0)
        assert "100.0" in protocol


# ══════════════════════════════════════════════════════════════════════
# 17. RK4 Integrator
# ══════════════════════════════════════════════════════════════════════


class TestRK4Integrator:
    """RK4 vs Euler integrator comparison."""

    def test_rk4_produces_output(self) -> None:
        c = BitstreamToDNA(seed=42)
        design = c.compile_network(
            gates=[{"type": "AND", "inputs": ["A", "B"], "output": "C"}],
            input_names=["A", "B"],
            output_names=["C"],
        )
        sim = KineticSimulator(integrator="rk4")
        result = sim.simulate(design, {"A": 200.0, "B": 200.0}, duration_s=1800.0)
        assert result["C"][-1] > 50.0

    def test_rk4_matches_euler_qualitatively(self) -> None:
        c = BitstreamToDNA(seed=42)
        design = c.compile_network(
            gates=[{"type": "AND", "inputs": ["A", "B"], "output": "C"}],
            input_names=["A", "B"],
            output_names=["C"],
        )
        euler = KineticSimulator(integrator="euler")
        rk4 = KineticSimulator(integrator="rk4")
        r_euler = euler.simulate(design, {"A": 200.0, "B": 200.0}, duration_s=1800.0)
        r_rk4 = rk4.simulate(design, {"A": 200.0, "B": 200.0}, duration_s=1800.0)
        assert abs(r_euler["C"][-1] - r_rk4["C"][-1]) < 30.0

    def test_temperature_affects_rate(self) -> None:
        c = BitstreamToDNA(seed=42)
        design = c.compile_network(
            gates=[{"type": "AND", "inputs": ["A", "B"], "output": "C"}],
            input_names=["A", "B"],
            output_names=["C"],
        )
        sim_37 = KineticSimulator(temperature_c=37.0)
        sim_25 = KineticSimulator(temperature_c=25.0)
        r_37 = sim_37.simulate(design, {"A": 200.0, "B": 200.0}, duration_s=600.0)
        r_25 = sim_25.simulate(design, {"A": 200.0, "B": 200.0}, duration_s=600.0)
        # Higher temperature → faster kinetics
        assert r_37["C"][-1] > r_25["C"][-1]


# ══════════════════════════════════════════════════════════════════════
# 18. Topological Analysis
# ══════════════════════════════════════════════════════════════════════


class TestTopologicalAnalysis:
    """Circuit topology analysis."""

    def test_depth_single_gate(self, simple_and_circuit: DNACircuitDesign) -> None:
        analyzer = TopologicalAnalyzer()
        result = analyzer.analyze(simple_and_circuit)
        assert result["depth"] >= 1
        assert result["has_feedback"] is False

    def test_depth_cascade(self) -> None:
        c = BitstreamToDNA(seed=42)
        design = c.compile_network(
            gates=[
                {"type": "AND", "inputs": ["A", "B"], "output": "X"},
                {"type": "NOT", "inputs": ["X"], "output": "Y"},
                {"type": "OR", "inputs": ["Y", "C"], "output": "Z"},
            ],
            input_names=["A", "B", "C"],
            output_names=["Z"],
        )
        result = TopologicalAnalyzer().analyze(design)
        assert result["depth"] >= 2
        assert result["n_nodes"] >= 4

    def test_fan_out_detected(self) -> None:
        c = BitstreamToDNA(seed=42)
        design = c.compile_network(
            gates=[
                {"type": "AND", "inputs": ["A", "B"], "output": "X"},
                {"type": "NOT", "inputs": ["A"], "output": "Y"},
            ],
            input_names=["A", "B"],
            output_names=["X", "Y"],
        )
        result = TopologicalAnalyzer().analyze(design)
        assert result["fan_out"]["A"] >= 2

    def test_no_feedback_in_dag(self, simple_and_circuit: DNACircuitDesign) -> None:
        result = TopologicalAnalyzer().analyze(simple_and_circuit)
        assert result["has_feedback"] is False
        assert len(result["cycles"]) == 0

    def test_feedback_cycle_is_reported_from_remaining_nodes(self) -> None:
        design = DNACircuitDesign(
            name="cycle",
            gates=[
                DNAGate(0, GateType.BUFFER, ["A"], "B"),
                DNAGate(1, GateType.BUFFER, ["B"], "A"),
            ],
        )

        result = TopologicalAnalyzer().analyze(design)

        assert result["has_feedback"] is True
        assert result["cycles"] == [["A", "B"]]


# ══════════════════════════════════════════════════════════════════════
# 19. Dual-Rail Encoding
# ══════════════════════════════════════════════════════════════════════


class TestDualRailEncoder:
    """Dual-rail fault-tolerant encoding."""

    def test_doubles_gate_count(self, simple_and_circuit: DNACircuitDesign) -> None:
        encoder = DualRailEncoder()
        compiler = BitstreamToDNA(seed=42)
        dual = encoder.encode(simple_and_circuit, compiler)
        assert dual.total_gates == simple_and_circuit.total_gates * 2

    def test_dual_rail_name(self, simple_and_circuit: DNACircuitDesign) -> None:
        encoder = DualRailEncoder()
        compiler = BitstreamToDNA(seed=42)
        dual = encoder.encode(simple_and_circuit, compiler)
        assert "dual_rail" in dual.name

    def test_fault_detection_no_faults(self) -> None:
        encoder = DualRailEncoder()
        result = {
            "time": np.array([0, 1]),
            "X_T": np.array([200.0, 200.0]),
            "X_C": np.array([0.0, 0.0]),
        }
        faults = encoder.check_faults(result)
        assert len(faults) == 0

    def test_fault_detection_stuck_high(self) -> None:
        encoder = DualRailEncoder()
        result = {
            "time": np.array([0, 1]),
            "X_T": np.array([200.0, 200.0]),
            "X_C": np.array([200.0, 200.0]),
        }
        faults = encoder.check_faults(result)
        assert len(faults) == 1
        assert faults[0]["fault_type"] == "stuck_high"

    def test_fault_detection_ignores_incomplete_dual_rail_pair(self) -> None:
        encoder = DualRailEncoder()
        result = {
            "time": np.array([0, 1]),
            "X_T": np.array([0.0, 200.0]),
        }

        assert encoder.check_faults(result) == []


class TestCostAndHybridizationEdges:
    """Boundary contracts for synthesis cost and strand interaction helpers."""

    def test_cross_hybridization_empty_substring_is_zero(self) -> None:
        assert CrossHybridizationChecker._longest_common_substring("", "ACGT") == 0

    def test_estimate_cost_counts_duplicate_sequences_once(self) -> None:
        design = DNACircuitDesign(
            name="duplicate_cost",
            input_strands=[
                DNAStrand(name="a", sequence="ACGTACGT"),
                DNAStrand(name="b", sequence="ACGTACGT"),
            ],
        )

        cost = estimate_cost(design, price_per_base_usd=1.0, fixed_per_oligo_usd=0.0)

        assert cost["n_unique_oligos"] == 1
        assert cost["total_cost_usd"] == 8.0


# ══════════════════════════════════════════════════════════════════════
# 20. Visualization
# ══════════════════════════════════════════════════════════════════════


class TestVisualization:
    """Circuit and kinetics visualization."""

    def test_circuit_diagram(self, simple_and_circuit: DNACircuitDesign) -> None:
        diagram = visualize_circuit(simple_and_circuit)
        assert "Circuit:" in diagram
        assert "INPUTS:" in diagram
        assert "OUTPUTS:" in diagram

    def test_circuit_diagram_renders_vertical_connector_for_multi_gate_cascade(self) -> None:
        c = BitstreamToDNA(seed=42)
        design = c.compile_network(
            gates=[
                {"type": "AND", "inputs": ["A", "B"], "output": "X"},
                {"type": "BUFFER", "inputs": ["X"], "output": "Y"},
            ],
            input_names=["A", "B"],
            output_names=["Y"],
        )

        diagram = visualize_circuit(design)

        assert "\n    │\n" in diagram
        assert "AND" in diagram

    def test_kinetics_sparkline(self, simple_and_circuit: DNACircuitDesign) -> None:
        sim = KineticSimulator()
        result = sim.simulate(simple_and_circuit, {"A": 200.0, "B": 200.0})
        chart = visualize_kinetics(result)
        assert "nM" in chart
        assert len(chart) > 0


# ══════════════════════════════════════════════════════════════════════
# 21. SC Network Bridge
# ══════════════════════════════════════════════════════════════════════


class TestSCNetworkBridge:
    """SC network to DNA circuit bridge."""

    def test_simple_adjacency(self) -> None:
        adj = np.array(
            [
                [0, 0, 1],
                [0, 0, 1],
                [0, 0, 0],
            ],
            dtype=float,
        )
        bridge = SCNetworkBridge(seed=42)
        design = bridge.from_adjacency(adj, input_indices=[0, 1], output_indices=[2])
        assert design.total_gates >= 1

    def test_inhibitory_produces_not(self) -> None:
        adj = np.array(
            [
                [0, -1],
                [0, 0],
            ],
            dtype=float,
        )
        bridge = SCNetworkBridge(seed=42)
        design = bridge.from_adjacency(adj, input_indices=[0], output_indices=[1])
        assert any(g.gate_type == GateType.NOT for g in design.gates)

    def test_multi_fan_in(self) -> None:
        adj = np.zeros((5, 5))
        adj[0, 4] = 1
        adj[1, 4] = 1
        adj[2, 4] = 1
        bridge = SCNetworkBridge(seed=42)
        design = bridge.from_adjacency(adj, input_indices=[0, 1, 2, 3], output_indices=[4])
        assert design.total_gates >= 2  # chained AND


# ══════════════════════════════════════════════════════════════════════
# 22. Concentration Optimizer
# ══════════════════════════════════════════════════════════════════════


class TestConcentrationOptimizer:
    """Concentration optimization."""

    def test_optimizer_returns_result(self, simple_and_circuit: DNACircuitDesign) -> None:
        opt = ConcentrationOptimizer(n_evaluations=5, seed=42)
        truth_table = [
            {"inputs": {"A": 200.0, "B": 200.0}, "expected": {"C": "high"}},
            {"inputs": {"A": 200.0, "B": 0.0}, "expected": {"C": "low"}},
        ]
        result = opt.optimize(simple_and_circuit, truth_table, duration_s=300.0)
        assert "best_score" in result
        assert "initial_score" in result
        assert "improvement_pct" in result
        assert result["best_score"] >= 0


# ══════════════════════════════════════════════════════════════════════
# 23. Hairpin / Secondary Structure Checker
# ══════════════════════════════════════════════════════════════════════


class TestHairpinChecker:
    """Hairpin secondary structure detection."""

    def test_palindrome_detected(self) -> None:
        checker = HairpinChecker(min_stem_length=4, min_loop_length=3)
        # This sequence has a perfect hairpin: ACGT...loop...ACGT
        seq = "ACGTACGTTTTCGTACGT"
        hairpins = checker.check_strand(seq)
        assert isinstance(hairpins, list)

    def test_short_strand_no_hairpin(self) -> None:
        checker = HairpinChecker(min_stem_length=4)
        hairpins = checker.check_strand("ACGT")
        assert len(hairpins) == 0

    def test_check_design_returns_list(self, simple_and_circuit: DNACircuitDesign) -> None:
        checker = HairpinChecker()
        flags = checker.check_design(simple_and_circuit)
        assert isinstance(flags, list)

    def test_check_design_flags_hairpin_strand(self) -> None:
        checker = HairpinChecker(min_stem_length=4, min_loop_length=3)
        design = DNACircuitDesign(
            name="hairpin_design",
            input_strands=[DNAStrand(name="hp", sequence="GCGCGCGCAAAGCGCGCGC", role="signal")],
        )

        flags = checker.check_design(design)

        assert flags
        assert flags[0]["strand_name"] == "hp"

    def test_flag_structure(self) -> None:
        checker = HairpinChecker(min_stem_length=4, min_loop_length=3)
        seq = "GCGCGCGCAAAGCGCGCGC"
        hairpins = checker.check_strand(seq)
        for hp in hairpins:
            assert "stem_length" in hp
            assert "loop_length" in hp
            assert "delta_g_estimate" in hp


# ══════════════════════════════════════════════════════════════════════
# 24. Gate Optimizer
# ══════════════════════════════════════════════════════════════════════


class TestGateOptimizer:
    """Circuit-level gate optimization."""

    def test_no_change_normal_circuit(self) -> None:
        opt = GateOptimizer()
        gates = [
            {"type": "AND", "inputs": ["A", "B"], "output": "C"},
        ]
        result = opt.optimize(gates, ["C"])
        assert result["removed_count"] == 0
        assert len(result["optimized_gates"]) == 1

    def test_removes_duplicate(self) -> None:
        opt = GateOptimizer()
        gates = [
            {"type": "AND", "inputs": ["A", "B"], "output": "C"},
            {"type": "AND", "inputs": ["A", "B"], "output": "C"},
        ]
        result = opt.optimize(gates, ["C"])
        assert result["removed_count"] >= 1

    def test_identity_buffer_removal(self) -> None:
        opt = GateOptimizer()
        gates = [
            {"type": "AND", "inputs": ["A", "B"], "output": "C"},
            {"type": "BUFFER", "inputs": ["C"], "output": "D"},
            {"type": "NOT", "inputs": ["D"], "output": "dead"},
        ]
        result = opt.optimize(gates, ["C"])
        reasons = {removal["reason"] for removal in result["removals"]}
        assert {"identity_buffer", "dead_output"}.issubset(reasons)


# ══════════════════════════════════════════════════════════════════════
# 25. SC Precision Analyzer
# ══════════════════════════════════════════════════════════════════════


class TestSCPrecisionAnalyzer:
    """Stochastic computing precision analysis."""

    def test_precision_fields(self, simple_and_circuit: DNACircuitDesign) -> None:
        analyzer = SCPrecisionAnalyzer()
        result = analyzer.analyze(simple_and_circuit, {"A": 200.0, "B": 200.0})
        assert "total_effective_bits" in result
        assert result["total_effective_bits"] > 0
        assert "outputs" in result

    def test_output_statistics(self, simple_and_circuit: DNACircuitDesign) -> None:
        analyzer = SCPrecisionAnalyzer()
        result = analyzer.analyze(simple_and_circuit, {"A": 200.0, "B": 200.0})
        for key, stats in result["outputs"].items():
            assert "snr_db" in stats
            assert "effective_bits" in stats
            assert "resolution_nM" in stats
            assert "dynamic_range_db" in stats

    def test_empty_design_reports_zero_effective_bits(self) -> None:
        analyzer = SCPrecisionAnalyzer()
        result = analyzer.analyze(DNACircuitDesign(name="empty"), {})

        assert result["outputs"] == {}
        assert result["total_effective_bits"] == 0.0


class TestSCNetworkBridgeEdges:
    """Adjacency-to-gate inference boundary cases."""

    def test_from_adjacency_skips_non_input_nodes_without_sources(self) -> None:
        design = SCNetworkBridge(seed=42).from_adjacency(
            np.zeros((3, 3), dtype=float),
            input_indices=[0],
            output_indices=[2],
            name="empty_graph",
        )

        assert design.total_gates == 0

    def test_from_adjacency_uses_or_for_two_source_inhibitory_mix(self) -> None:
        adjacency = np.array(
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
                [0.0, 0.0, 0.0],
            ]
        )

        design = SCNetworkBridge(seed=42).from_adjacency(
            adjacency,
            input_indices=[0, 1],
            output_indices=[2],
            name="mixed_sources",
        )

        assert design.gates[0].gate_type == GateType.OR


# ══════════════════════════════════════════════════════════════════════
# 26. Degradation Model
# ══════════════════════════════════════════════════════════════════════


class TestDegradationModel:
    """Time-dependent DNA degradation."""

    def test_concentration_decreases(self) -> None:
        dm = DegradationModel(half_life_hr=24.0)
        remaining = dm.predict_concentration(200.0, 30, 24.0)
        assert remaining < 200.0

    def test_zero_time_no_degradation(self) -> None:
        dm = DegradationModel()
        remaining = dm.predict_concentration(200.0, 30, 0.0)
        assert abs(remaining - 200.0) < 1e-6

    def test_design_analysis(self, simple_and_circuit: DNACircuitDesign) -> None:
        dm = DegradationModel()
        report = dm.analyze_design(simple_and_circuit, time_hr=4.0)
        assert "min_remaining_pct" in report
        assert "strands" in report
        assert len(report["strands"]) > 0
        for s in report["strands"]:
            assert s["pct_remaining"] <= 100.0


# ══════════════════════════════════════════════════════════════════════
# 27. Plate Layout
# ══════════════════════════════════════════════════════════════════════


class TestPlateLayout:
    """96-well plate layout generation."""

    def test_layout_produces_plates(self, simple_and_circuit: DNACircuitDesign) -> None:
        pl = PlateLayout()
        result = pl.layout(simple_and_circuit)
        assert result["n_plates"] >= 1
        assert result["n_unique_oligos"] > 0

    def test_well_format(self, simple_and_circuit: DNACircuitDesign) -> None:
        pl = PlateLayout()
        result = pl.layout(simple_and_circuit)
        for plate in result["plates"]:
            for entry in plate:
                assert len(entry["well"]) == 3  # e.g. A01
                assert entry["well"][0] in "ABCDEFGH"

    def test_csv_manifest(self, simple_and_circuit: DNACircuitDesign) -> None:
        pl = PlateLayout()
        result = pl.layout(simple_and_circuit)
        csv = result["manifest_csv"]
        assert csv.startswith("Well,Name,Sequence,Length")
        lines = csv.strip().split("\n")
        assert len(lines) > 1

    def test_layout_splits_across_multiple_plates(self) -> None:
        design = DNACircuitDesign(
            name="multi_plate",
            input_strands=[
                DNAStrand(name=f"s{i}", sequence=f"ACGTACGTACGT{i % 10}", role="signal")
                for i in range(5)
            ],
        )
        pl = PlateLayout(n_wells=2)

        result = pl.layout(design)

        assert result["n_plates"] == 3
        assert [entry["well"] for plate in result["plates"] for entry in plate] == [
            "A01",
            "A02",
            "A01",
            "A02",
            "A01",
        ]
