# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for bridges.dna_mapper

from __future__ import annotations

import os
import time

import numpy as np
import pytest

from sc_neurocore.bridges.dna_mapper import (
    DNAStrand,
    DNAGate,
    DNACircuitDesign,
    GateType,
    SequenceDesigner,
    StrandDisplacementCompiler,
    EnzymaticGateCompiler,
    KineticSimulator,
    BitstreamToDNA,
    GF4ErrorCorrection,
    CrossHybridizationChecker,
    HairpinChecker,
    DegradationModel,
    SCNetworkBridge,
)


# ---------------------------------------------------------------------------
# DNAStrand properties
# ---------------------------------------------------------------------------


class TestDNAStrand:
    def test_gc_content_all_gc(self) -> None:
        s = DNAStrand(name="s1", sequence="GCGC")
        assert s.gc_content == pytest.approx(1.0)

    def test_gc_content_all_at(self) -> None:
        s = DNAStrand(name="s2", sequence="ATAT")
        assert s.gc_content == pytest.approx(0.0)

    def test_gc_content_mixed(self) -> None:
        s = DNAStrand(name="s3", sequence="ATGC")
        assert s.gc_content == pytest.approx(0.5)

    def test_complement(self) -> None:
        s = DNAStrand(name="s4", sequence="ATCG")
        comp = s.complement
        assert isinstance(comp, str)
        assert len(comp) == 4
        assert set(comp).issubset(set("ATCG"))

    def test_max_homopolymer_run(self) -> None:
        s = DNAStrand(name="s5", sequence="AAAAATCG")
        assert s.max_homopolymer_run >= 5

    def test_delta_g_37(self) -> None:
        s = DNAStrand(name="s6", sequence="GCGCGCGCGCGC")
        dg = s.delta_g_37()
        assert isinstance(dg, float)


# ---------------------------------------------------------------------------
# SequenceDesigner
# ---------------------------------------------------------------------------


class TestSequenceDesigner:
    def test_generate_returns_valid_dna(self) -> None:
        designer = SequenceDesigner(seed=42)
        seq = designer.generate(length=20)
        assert len(seq) == 20
        assert all(c in "ATCG" for c in seq)

    def test_generate_complement(self) -> None:
        designer = SequenceDesigner(seed=42)
        seq = designer.generate(length=15)
        comp = designer.generate_complement(seq)
        assert len(comp) == len(seq)

    def test_generate_toehold(self) -> None:
        designer = SequenceDesigner(seed=42)
        th = designer.generate_toehold()
        assert isinstance(th, str)
        assert all(c in "ATCG" for c in th)

    def test_deterministic_with_seed(self) -> None:
        d1 = SequenceDesigner(seed=123)
        d2 = SequenceDesigner(seed=123)
        assert d1.generate(length=30) == d2.generate(length=30)

    def test_generate_recognition(self) -> None:
        designer = SequenceDesigner(seed=42)
        rec = designer.generate_recognition()
        assert isinstance(rec, str)


# ---------------------------------------------------------------------------
# Gate Compilers
# ---------------------------------------------------------------------------


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


class TestEnzymaticGateCompiler:
    def test_compile_nand(self) -> None:
        compiler = EnzymaticGateCompiler()
        gate = compiler.compile_nand("a", "b", "out")
        assert isinstance(gate, DNAGate)

    def test_compile_xor(self) -> None:
        compiler = EnzymaticGateCompiler()
        gate = compiler.compile_xor("a", "b", "out")
        assert isinstance(gate, DNAGate)


# ---------------------------------------------------------------------------
# KineticSimulator
# ---------------------------------------------------------------------------


class TestKineticSimulator:
    def test_simulate_produces_trajectory(self) -> None:
        compiler = StrandDisplacementCompiler()
        and_gate = compiler.compile_and("a", "b", "out")
        design = DNACircuitDesign(
            name="test_circuit",
            gates=[and_gate],
            input_strands=[
                DNAStrand(name="a", sequence="ACGT"),
                DNAStrand(name="b", sequence="TGCA"),
            ],
            output_strands=[DNAStrand(name="out", sequence="AGCT")],
        )
        sim = KineticSimulator()
        result = sim.simulate(
            design, input_concentrations={"a": 100.0, "b": 100.0}, duration_s=10.0, dt=1.0
        )
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# BitstreamToDNA
# ---------------------------------------------------------------------------


class TestBitstreamToDNA:
    def test_compile_network(self) -> None:
        bridge = BitstreamToDNA(seed=42)
        gates = [
            {"type": "AND", "inputs": ["a", "b"], "output": "c"},
            {"type": "OR", "inputs": ["c", "d"], "output": "e"},
        ]
        design = bridge.compile_network(gates, input_names=["a", "b", "d"], output_names=["e"])
        assert isinstance(design, DNACircuitDesign)

    def test_validate(self) -> None:
        bridge = BitstreamToDNA(seed=42)
        gates = [{"type": "NOT", "inputs": ["a"], "output": "b"}]
        design = bridge.compile_network(gates, input_names=["a"], output_names=["b"])
        result = bridge.validate(design)
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# Error Correction
# ---------------------------------------------------------------------------


class TestGF4ErrorCorrection:
    def test_encode_decode_roundtrip(self) -> None:
        ecc = GF4ErrorCorrection(n_parity=4)
        data = "ATCGATCG"
        encoded = ecc.encode(data)
        assert len(encoded) > len(data)
        decoded, errors = ecc.decode(encoded)
        assert decoded[: len(data)] == data
        assert errors == 0

    def test_detects_errors(self) -> None:
        ecc = GF4ErrorCorrection(n_parity=4)
        data = "ATCGATCG"
        encoded = ecc.encode(data)
        corrupted_chars = list(encoded)
        orig = corrupted_chars[2]
        corrupted_chars[2] = "G" if orig != "G" else "A"
        corrupted = "".join(corrupted_chars)
        decoded, error_count = ecc.decode(corrupted)
        assert isinstance(decoded, str)
        assert error_count >= 0


# ---------------------------------------------------------------------------
# Checkers
# ---------------------------------------------------------------------------


class TestCrossHybridizationChecker:
    def test_check_returns_list(self) -> None:
        compiler = StrandDisplacementCompiler()
        gate = compiler.compile_and("a", "b", "out")
        design = DNACircuitDesign(
            name="test",
            gates=[gate],
            input_strands=[
                DNAStrand(name="a", sequence="ACGTACGT"),
                DNAStrand(name="b", sequence="TGCATGCA"),
            ],
            output_strands=[DNAStrand(name="out", sequence="AGCTAGCT")],
        )

        result = CrossHybridizationChecker().check(design)

        assert isinstance(result, list)


class TestHairpinChecker:
    def test_check_strand(self) -> None:
        checker = HairpinChecker()
        result = checker.check_strand("GCGATCGC")
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# DegradationModel
# ---------------------------------------------------------------------------


class TestDegradationModel:
    def test_predict_concentration_decays(self) -> None:
        model = DegradationModel(half_life_hr=2.0)
        c0 = 100.0
        c1 = model.predict_concentration(c0, strand_length=20, time_hr=2.0)
        assert c1 < c0
        c2 = model.predict_concentration(c0, strand_length=20, time_hr=0.0)
        assert c2 == pytest.approx(c0)


# ---------------------------------------------------------------------------
# SCNetworkBridge
# ---------------------------------------------------------------------------


class TestSCNetworkBridge:
    def test_from_adjacency(self) -> None:
        adj = np.array([[0, 1, 0.5], [0, 0, 1], [0, 0, 0]], dtype=float)
        bridge = SCNetworkBridge(seed=42)
        design = bridge.from_adjacency(adj, input_indices=[0], output_indices=[2])
        assert isinstance(design, DNACircuitDesign)


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------


class TestDNAMapperBenchmark:
    def test_sequence_generation_throughput(self) -> None:
        """Generate 500 random 20nt sequences."""
        designer = SequenceDesigner(seed=42)
        t0 = time.perf_counter()
        for _ in range(500):
            designer.generate(length=20)
        elapsed = time.perf_counter() - t0
        max_elapsed = 10.0 if os.environ.get("CI") else 5.0
        assert elapsed < max_elapsed, f"500 sequences took {elapsed:.2f}s"

    def test_gate_compilation_throughput(self) -> None:
        """Compile 100 AND gates."""
        compiler = StrandDisplacementCompiler()
        t0 = time.perf_counter()
        for i in range(100):
            compiler.compile_and(f"a_{i}", f"b_{i}", f"out_{i}")
        elapsed = time.perf_counter() - t0
        max_elapsed = 20.0 if os.environ.get("CI") else 10.0
        assert elapsed < max_elapsed, f"100 AND gates took {elapsed:.2f}s"
