# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for bridges.photonic_noc

from __future__ import annotations

import math
import time

import numpy as np
import pytest

from sc_neurocore.bridges.photonic_noc import (
    WaveguideSegment,
    MZIGate,
    WDMChannel,
    PhotonicCircuitDesign,
    WaveguideRouter,
    MZICompiler,
    WDMAssigner,
    PowerBudgetAnalyzer,
    SCToPhotonic,
    ThermalPhaseShifter,
    CrosstalkAnalyzer,
)


# ---------------------------------------------------------------------------
# Waveguide Routing
# ---------------------------------------------------------------------------


class TestWaveguideRouter:
    def test_route_two_nodes(self):
        router = WaveguideRouter()
        adj = np.array([[0, 1], [0, 0]], dtype=float)
        segments = router.route(adj)
        assert isinstance(segments, list)
        assert len(segments) >= 1
        assert all(isinstance(s, WaveguideSegment) for s in segments)

    def test_route_triangle(self):
        router = WaveguideRouter()
        adj = np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]], dtype=float)
        segments = router.route(adj)
        assert len(segments) >= 3

    def test_no_self_loops(self):
        router = WaveguideRouter()
        adj = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
        segments = router.route(adj)
        for seg in segments:
            assert seg.source != seg.target

    def test_route_empty_graph(self):
        router = WaveguideRouter()
        adj = np.zeros((5, 5), dtype=float)
        segments = router.route(adj)
        assert len(segments) == 0

    def test_segment_positive_length(self):
        router = WaveguideRouter()
        adj = np.array([[0, 1], [0, 0]], dtype=float)
        segments = router.route(adj)
        for seg in segments:
            assert seg.length_um > 0


# ---------------------------------------------------------------------------
# MZI Compiler
# ---------------------------------------------------------------------------


class TestMZICompiler:
    def test_compile_gate(self):
        compiler = MZICompiler()
        gate = compiler.compile_gate(gate_type="cross", input_ports=[0, 1], output_port=0)
        assert isinstance(gate, MZIGate)

    def test_compile_network(self):
        compiler = MZICompiler()
        gates = [
            {"type": "cross", "inputs": [0, 1], "output": 0},
            {"type": "bar", "inputs": [1, 2], "output": 1},
        ]
        result = compiler.compile_network(gates)
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_phase_shift_range(self):
        compiler = MZICompiler()
        gate = compiler.compile_gate(gate_type="bar", input_ports=[0, 1], output_port=1)
        assert 0 <= gate.phase_shift_rad <= 2 * math.pi


# ---------------------------------------------------------------------------
# WDM Assigner
# ---------------------------------------------------------------------------


class TestWDMAssigner:
    def test_assign_no_conflicts(self):
        assigner = WDMAssigner(max_channels=8)
        channels = assigner.assign(["sig_a", "sig_b", "sig_c"])
        assert isinstance(channels, list)
        assert len(channels) == 3
        wavelengths = [ch.wavelength_nm for ch in channels]
        assert len(set(wavelengths)) == 3

    def test_channel_wavelength_positive(self):
        assigner = WDMAssigner()
        channels = assigner.assign(["s1", "s2"])
        for ch in channels:
            assert ch.wavelength_nm > 0
            assert ch.bandwidth_nm > 0


# ---------------------------------------------------------------------------
# Power Budget Analyzer
# ---------------------------------------------------------------------------


class TestPowerBudgetAnalyzer:
    def test_analyze_returns_dict(self):
        compiler = SCToPhotonic()
        adj = np.array([[0, 1], [0, 0]], dtype=float)
        design = compiler.compile(adj)
        analyzer = PowerBudgetAnalyzer()
        result = analyzer.analyze(design)
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# SCToPhotonic end-to-end
# ---------------------------------------------------------------------------


class TestSCToPhotonic:
    def test_compile_simple_network(self):
        compiler = SCToPhotonic()
        adj = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=float)
        design = compiler.compile(adj)
        assert isinstance(design, PhotonicCircuitDesign)
        assert design.n_nodes >= 3
        assert len(design.waveguides) >= 1
        assert len(design.mzi_gates) >= 1

    def test_compile_larger_network(self):
        compiler = SCToPhotonic()
        rng = np.random.default_rng(42)
        adj = (rng.random((10, 10)) > 0.7).astype(float)
        np.fill_diagonal(adj, 0)
        design = compiler.compile(adj)
        assert design.n_nodes == 10

    def test_design_has_wdm_channels(self):
        compiler = SCToPhotonic()
        adj = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)
        design = compiler.compile(adj)
        assert len(design.wdm_channels) >= 1


# ---------------------------------------------------------------------------
# Thermal Phase Shifter
# ---------------------------------------------------------------------------


class TestThermalPhaseShifter:
    def test_power_for_pi_phase_positive(self):
        shifter = ThermalPhaseShifter()
        power = shifter.power_for_phase(math.pi)
        assert power > 0

    def test_power_for_zero_phase(self):
        shifter = ThermalPhaseShifter()
        power = shifter.power_for_phase(0.0)
        assert power == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Crosstalk Analyzer
# ---------------------------------------------------------------------------


class TestCrosstalkAnalyzer:
    def test_analyze_returns_result(self):
        analyzer = CrosstalkAnalyzer()
        ch1 = WDMChannel(channel_id=0, wavelength_nm=1550.0, bandwidth_nm=0.4, signal_name="s1")
        ch2 = WDMChannel(channel_id=1, wavelength_nm=1550.8, bandwidth_nm=0.4, signal_name="s2")
        result = analyzer.analyze([ch1, ch2])
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------


class TestPhotonicBenchmark:
    def test_compile_50_node_network(self):
        """End-to-end 50-node SC → photonic compile."""
        compiler = SCToPhotonic()
        rng = np.random.default_rng(42)
        adj = (rng.random((50, 50)) > 0.85).astype(float)
        np.fill_diagonal(adj, 0)
        t0 = time.perf_counter()
        design = compiler.compile(adj)
        elapsed = time.perf_counter() - t0
        assert design.n_nodes == 50
        assert elapsed < 10.0, f"50-node photonic compile took {elapsed:.1f}s"

    def test_wdm_assignment_throughput(self):
        """WDM assignment for 100 signals."""
        assigner = WDMAssigner(max_channels=128)
        signals = [f"sig_{i}" for i in range(100)]
        t0 = time.perf_counter()
        channels = assigner.assign(signals)
        elapsed = time.perf_counter() - t0
        assert len(channels) == 100
        assert elapsed < 1.0, f"100-signal WDM took {elapsed:.2f}s"
