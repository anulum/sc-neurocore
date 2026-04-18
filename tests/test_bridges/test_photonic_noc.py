# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

"""Tests for sc_neurocore.bridges.photonic_noc."""

from __future__ import annotations

import json
import math
import os

import numpy as np
import pytest

from sc_neurocore.bridges.photonic_noc import (
    CrosstalkAnalyzer,
    MZICompiler,
    MZIGate,
    PhotonicCircuitDesign,
    PowerBudgetAnalyzer,
    SCToPhotonic,
    ThermalPhaseShifter,
    WDMAssigner,
    WDMChannel,
    WaveguideRouter,
    WaveguideSegment,
    WaveguideType,
    export_photonic_json,
    visualize_photonic,
)


# ══════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════


@pytest.fixture
def simple_adjacency() -> np.ndarray:
    """4-node mesh network."""
    return np.array(
        [
            [0.0, 1.0, 0.5, 0.0],
            [1.0, 0.0, 0.0, 0.8],
            [0.5, 0.0, 0.0, 1.0],
            [0.0, 0.8, 1.0, 0.0],
        ]
    )


@pytest.fixture
def simple_design(simple_adjacency: np.ndarray) -> PhotonicCircuitDesign:
    """Compiled 4-node photonic design."""
    compiler = SCToPhotonic()
    return compiler.compile(simple_adjacency, name="test_noc")


# ══════════════════════════════════════════════════════════════════════
# 1. Data Types
# ══════════════════════════════════════════════════════════════════════


class TestDataTypes:
    """Photonic data class tests."""

    def test_waveguide_type_enum(self) -> None:
        assert WaveguideType.STRIP.value == "strip"
        assert WaveguideType.RIB.value == "rib"

    def test_waveguide_segment(self) -> None:
        wg = WaveguideSegment(source=0, target=1, length_um=500.0, loss_db=1.0)
        assert wg.source == 0
        assert wg.length_um == 500.0

    def test_mzi_gate(self) -> None:
        mzi = MZIGate(gate_id="mzi_0", operation="MUL", phase_shift_rad=math.pi / 2)
        assert mzi.operation == "MUL"
        assert abs(mzi.phase_shift_rad - math.pi / 2) < 1e-10

    def test_wdm_channel(self) -> None:
        ch = WDMChannel(channel_id=0, wavelength_nm=1550.0, signal_name="a")
        assert ch.wavelength_nm == 1550.0


# ══════════════════════════════════════════════════════════════════════
# 2. Waveguide Router
# ══════════════════════════════════════════════════════════════════════


class TestWaveguideRouter:
    """Waveguide routing tests."""

    def test_route_produces_segments(self, simple_adjacency: np.ndarray) -> None:
        router = WaveguideRouter()
        segments = router.route(simple_adjacency)
        assert len(segments) > 0
        assert all(isinstance(s, WaveguideSegment) for s in segments)

    def test_loss_positive(self, simple_adjacency: np.ndarray) -> None:
        router = WaveguideRouter()
        segments = router.route(simple_adjacency)
        for s in segments:
            assert s.loss_db >= 0

    def test_no_self_loops(self, simple_adjacency: np.ndarray) -> None:
        router = WaveguideRouter()
        segments = router.route(simple_adjacency)
        for s in segments:
            assert s.source != s.target

    def test_custom_pitch(self) -> None:
        adj = np.array([[0.0, 1.0], [1.0, 0.0]])
        r1 = WaveguideRouter(pitch_um=100.0).route(adj)
        r2 = WaveguideRouter(pitch_um=500.0).route(adj)
        assert r1[0].length_um < r2[0].length_um


# ══════════════════════════════════════════════════════════════════════
# 3. MZI Compiler
# ══════════════════════════════════════════════════════════════════════


class TestMZICompiler:
    """MZI gate compilation tests."""

    def test_compile_and_gate(self) -> None:
        mzi = MZICompiler().compile_gate("AND", [0, 1], 2)
        assert mzi.operation == "AND"
        assert abs(mzi.phase_shift_rad - math.pi / 2) < 1e-10

    def test_compile_not_gate(self) -> None:
        mzi = MZICompiler().compile_gate("NOT", [0], 1)
        assert abs(mzi.phase_shift_rad - math.pi) < 1e-10

    def test_compile_network(self) -> None:
        gates = [
            {"type": "MUL", "inputs": [0, 1], "output": 2},
            {"type": "ADD", "inputs": [2, 3], "output": 4},
        ]
        mzis = MZICompiler().compile_network(gates)
        assert len(mzis) == 2


# ══════════════════════════════════════════════════════════════════════
# 4. WDM Assigner
# ══════════════════════════════════════════════════════════════════════


class TestWDMAssigner:
    """WDM channel assignment tests."""

    def test_assign_channels(self) -> None:
        assigner = WDMAssigner()
        channels = assigner.assign(["a", "b", "c"])
        assert len(channels) == 3

    def test_wavelength_spacing(self) -> None:
        assigner = WDMAssigner(channel_spacing_nm=0.8)
        channels = assigner.assign(["a", "b"])
        assert abs(channels[1].wavelength_nm - channels[0].wavelength_nm - 0.8) < 1e-10

    def test_signal_names(self) -> None:
        channels = WDMAssigner().assign(["x", "y"])
        assert channels[0].signal_name == "x"
        assert channels[1].signal_name == "y"

    # --- max_channels cap (closes task #47) ---

    def test_default_max_channels_is_96(self) -> None:
        a = WDMAssigner()
        assert a._max_channels == 96

    def test_assign_at_default_cap_succeeds(self) -> None:
        names = [f"sig{i}" for i in range(96)]
        channels = WDMAssigner().assign(names)
        assert len(channels) == 96

    def test_assign_above_default_cap_raises(self) -> None:
        import pytest

        names = [f"sig{i}" for i in range(97)]
        with pytest.raises(ValueError, match="max_channels"):
            WDMAssigner().assign(names)

    def test_explicit_smaller_cap_raises(self) -> None:
        import pytest

        names = ["a", "b", "c"]
        with pytest.raises(ValueError, match="max_channels"):
            WDMAssigner(max_channels=2).assign(names)

    def test_max_channels_zero_disables_cap(self) -> None:
        names = [f"sig{i}" for i in range(200)]
        channels = WDMAssigner(max_channels=0).assign(names)
        assert len(channels) == 200


# ══════════════════════════════════════════════════════════════════════
# 5. Power Budget Analyzer
# ══════════════════════════════════════════════════════════════════════


class TestPowerBudgetAnalyzer:
    """Power budget analysis tests."""

    def test_analyze(self, simple_design: PhotonicCircuitDesign) -> None:
        pba = PowerBudgetAnalyzer()
        result = pba.analyze(simple_design)
        assert result["n_paths"] > 0
        assert "worst_margin_db" in result

    def test_all_paths_have_margin(self, simple_design: PhotonicCircuitDesign) -> None:
        result = PowerBudgetAnalyzer().analyze(simple_design)
        for path in result["paths"]:
            assert "margin_db" in path
            assert "passed" in path


# ══════════════════════════════════════════════════════════════════════
# 6. SCToPhotonic Compiler
# ══════════════════════════════════════════════════════════════════════


class TestSCToPhotonic:
    """Top-level photonic compiler tests."""

    def test_compile_basic(self, simple_adjacency: np.ndarray) -> None:
        design = SCToPhotonic().compile(simple_adjacency, name="test")
        assert design.n_nodes == 4
        assert len(design.waveguides) > 0
        assert len(design.mzi_gates) > 0
        assert len(design.wdm_channels) == 4

    def test_with_labels(self, simple_adjacency: np.ndarray) -> None:
        design = SCToPhotonic().compile(simple_adjacency, node_labels=["A", "B", "C", "D"])
        assert design.wdm_channels[0].signal_name == "A"

    def test_area_positive(self, simple_adjacency: np.ndarray) -> None:
        design = SCToPhotonic().compile(simple_adjacency)
        assert design.total_area_um2 > 0

    def test_with_custom_gates(self, simple_adjacency: np.ndarray) -> None:
        gates = [{"type": "MUL", "inputs": [0, 1], "output": 2}]
        design = SCToPhotonic().compile(simple_adjacency, gate_specs=gates)
        assert len(design.mzi_gates) == 1


# ══════════════════════════════════════════════════════════════════════
# 7. Thermal Phase Shifter
# ══════════════════════════════════════════════════════════════════════


class TestThermalPhaseShifter:
    """Thermal tuning model tests."""

    def test_power_for_pi(self) -> None:
        tps = ThermalPhaseShifter()
        p = tps.power_for_phase(math.pi)
        assert p > 0

    def test_power_scales_with_phase(self) -> None:
        tps = ThermalPhaseShifter()
        p1 = tps.power_for_phase(math.pi / 4)
        p2 = tps.power_for_phase(math.pi / 2)
        assert p2 > p1

    def test_analyze_design(self, simple_design: PhotonicCircuitDesign) -> None:
        tps = ThermalPhaseShifter()
        result = tps.analyze_design(simple_design)
        assert result["total_power_mw"] > 0
        assert result["n_gates"] == len(simple_design.mzi_gates)


# ══════════════════════════════════════════════════════════════════════
# 8. Crosstalk Analyzer
# ══════════════════════════════════════════════════════════════════════


class TestCrosstalkAnalyzer:
    """WDM crosstalk tests."""

    def test_analyze(self, simple_design: PhotonicCircuitDesign) -> None:
        ct = CrosstalkAnalyzer()
        result = ct.analyze(simple_design.wdm_channels)
        assert result["n_channels"] == 4
        assert "worst_xt_db" in result

    def test_per_channel_osnr(self, simple_design: PhotonicCircuitDesign) -> None:
        result = CrosstalkAnalyzer().analyze(simple_design.wdm_channels)
        for ch in result["per_channel"]:
            assert "osnr_db" in ch


# ══════════════════════════════════════════════════════════════════════
# 9. Export & Visualization
# ══════════════════════════════════════════════════════════════════════


class TestExportVisualization:
    """Export and visualization tests."""

    def test_export_json(self, simple_design: PhotonicCircuitDesign, tmp_path: str) -> None:
        path = os.path.join(tmp_path, "photonic.json")
        export_photonic_json(simple_design, path)
        with open(path) as f:
            data = json.load(f)
        assert data["n_nodes"] == 4
        assert len(data["waveguides"]) > 0

    def test_visualize(self, simple_design: PhotonicCircuitDesign) -> None:
        viz = visualize_photonic(simple_design)
        assert "Photonic NoC" in viz
        assert "Waveguides" in viz
        assert "MZI Gates" in viz
