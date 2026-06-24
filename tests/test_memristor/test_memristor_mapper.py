# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Memristor Mapper Tests

import numpy as np
import pytest

from sc_neurocore.memristor.memristor_mapper import (
    AgingSimulator,
    CompensationLUT,
    CompensationStrategy,
    ConductanceModel,
    CrossbarArray,
    CrossbarEstimator,
    CrossbarTopology,
    IRDropModel,
    MemristorMapper,
    MemristorTechnology,
    MonteCarloSimulator,
    SCAbsorbEncoder,
    SneakPathModel,
    StuckFaultMap,
    VariabilityInjector,
    VerilogEmitter,
    WriteVerifyProtocol,
)


# ── ConductanceModel Tests ──────────────────────────────────────────


class TestConductanceModel:
    def test_defaults_from_technology(self):
        m = ConductanceModel(MemristorTechnology.RERAM_HFOX)
        assert m.g_on == 100e-6
        assert m.g_off == 1e-6
        assert m.sigma_g == 0.05

    def test_dynamic_range(self):
        m = ConductanceModel(MemristorTechnology.RERAM_HFOX)
        assert m.dynamic_range == pytest.approx(100.0)

    def test_level_step_positive(self):
        m = ConductanceModel(MemristorTechnology.GENERIC)
        assert m.level_step > 0

    def test_target_conductance_bounds(self):
        m = ConductanceModel(MemristorTechnology.PCM)
        assert m.target_conductance(0) == m.g_off
        assert m.target_conductance(m.num_levels - 1) == pytest.approx(m.g_on)

    def test_target_conductance_clamps(self):
        m = ConductanceModel(MemristorTechnology.GENERIC)
        assert m.target_conductance(-1) == m.g_off
        assert m.target_conductance(9999) == pytest.approx(m.g_on)

    def test_sample_d2d_different_each_call(self):
        m = ConductanceModel(MemristorTechnology.RERAM_HFOX)
        rng = np.random.default_rng(42)
        s1 = m.sample_d2d(8, rng)
        s2 = m.sample_d2d(8, rng)
        assert s1 != s2

    def test_sample_rw_adds_noise(self):
        m = ConductanceModel(MemristorTechnology.GENERIC)
        rng = np.random.default_rng(42)
        vals = [m.sample_rw(50e-6, rng) for _ in range(100)]
        assert np.std(vals) > 0

    def test_all_technologies_load(self):
        for tech in MemristorTechnology:
            m = ConductanceModel(tech)
            assert m.g_on > m.g_off
            assert m.num_levels >= 1

    def test_mythic_high_levels(self):
        m = ConductanceModel(MemristorTechnology.MYTHIC_AMP)
        assert m.num_levels == 256

    def test_2d_material_higher_endurance_model(self):
        m = ConductanceModel(MemristorTechnology.RERAM_2D)
        assert m.sigma_g == 0.08


# ── CrossbarArray Tests ──────────────────────────────────────────────


class TestCrossbarArray:
    def test_num_devices_standard(self):
        xbar = CrossbarArray(64, 64)
        assert xbar.num_devices == 4096

    def test_num_devices_differential(self):
        xbar = CrossbarArray(32, 32, CrossbarTopology.DIFFERENTIAL)
        assert xbar.num_devices == 2048

    def test_conductance_model(self):
        xbar = CrossbarArray(16, 16, technology=MemristorTechnology.PCM)
        m = xbar.conductance_model
        assert m.technology == MemristorTechnology.PCM


# ── VariabilityInjector Tests ────────────────────────────────────────


class TestVariabilityInjector:
    def test_quantize_weights(self):
        m = ConductanceModel(MemristorTechnology.GENERIC)
        inj = VariabilityInjector(m, seed=42)
        w = np.array([[0.0, 0.5, 1.0]])
        levels = inj.quantize_weights(w)
        assert levels[0, 0] == 0
        assert levels[0, 2] == m.num_levels - 1

    def test_inject_d2d_changes_values(self):
        m = ConductanceModel(MemristorTechnology.RERAM_HFOX)
        inj = VariabilityInjector(m, seed=42)
        levels = np.array([[8, 8, 8]])
        g = inj.inject_d2d(levels)
        assert not np.all(g == g[0, 0])

    def test_inject_rw_adds_noise(self):
        m = ConductanceModel(MemristorTechnology.GENERIC)
        inj = VariabilityInjector(m, seed=42)
        g = np.full((4, 4), 50e-6)
        noisy = inj.inject_rw(g)
        assert not np.allclose(g, noisy)

    def test_inject_full_pipeline(self):
        m = ConductanceModel(MemristorTechnology.GENERIC)
        inj = VariabilityInjector(m, seed=42)
        w = np.random.default_rng(0).random((4, 4))
        levels, cond = inj.inject_full(w)
        assert levels.shape == (4, 4)
        assert cond.shape == (4, 4)
        assert np.all(levels >= 0)
        assert np.all(levels < m.num_levels)

    def test_compute_error_positive(self):
        m = ConductanceModel(MemristorTechnology.GENERIC)
        inj = VariabilityInjector(m, seed=42)
        w = np.random.default_rng(0).random((4, 4))
        levels, cond = inj.inject_full(w)
        err = inj.compute_error(w, cond)
        assert err["mae"] >= 0
        assert err["mean_rel_err"] >= 0

    def test_deterministic_with_same_seed(self):
        m = ConductanceModel(MemristorTechnology.GENERIC)
        w = np.random.default_rng(0).random((3, 3))
        inj1 = VariabilityInjector(m, seed=42)
        _, g1 = inj1.inject_full(w)
        inj2 = VariabilityInjector(m, seed=42)
        _, g2 = inj2.inject_full(w)
        np.testing.assert_array_equal(g1, g2)


# ── CompensationLUT Tests ────────────────────────────────────────────


class TestCompensationLUT:
    def test_build_nominal(self):
        m = ConductanceModel(MemristorTechnology.GENERIC)
        lut = CompensationLUT.build((0, 0), m)
        assert len(lut.compensated_thresholds) == m.num_levels

    def test_nominal_no_compensation(self):
        m = ConductanceModel(MemristorTechnology.GENERIC)
        lut = CompensationLUT.build((0, 0), m)
        assert lut.max_compensation < 0.01

    def test_measured_applies_compensation(self):
        m = ConductanceModel(MemristorTechnology.GENERIC)
        measured = np.array([m.target_conductance(i) * 0.9 for i in range(m.num_levels)])
        lut = CompensationLUT.build((0, 0), m, measured)
        assert lut.max_compensation > 0.05

    def test_device_id_stored(self):
        m = ConductanceModel(MemristorTechnology.GENERIC)
        lut = CompensationLUT.build((3, 7), m)
        assert lut.device_id == (3, 7)


# ── MemristorMapper Tests ───────────────────────────────────────────


class TestMemristorMapper:
    def test_map_small_matrix(self):
        mapper = MemristorMapper(seed=42)
        w = np.random.default_rng(0).random((8, 8))
        result = mapper.map_weights(w)
        assert result.total_crossbars == 1
        assert result.total_devices == 64

    def test_map_tiled(self):
        mapper = MemristorMapper(max_crossbar_size=4, seed=42)
        w = np.random.default_rng(0).random((8, 8))
        result = mapper.map_weights(w)
        assert result.total_crossbars == 4

    def test_map_1d_vector(self):
        mapper = MemristorMapper(seed=42)
        w = np.random.default_rng(0).random(16)
        result = mapper.map_weights(w)
        assert result.total_crossbars == 1

    def test_error_stats_present(self):
        mapper = MemristorMapper(seed=42)
        w = np.random.default_rng(0).random((4, 4))
        result = mapper.map_weights(w)
        assert result.mean_rel_error >= 0
        assert result.max_rel_error >= 0

    def test_compensation_luts_generated(self):
        mapper = MemristorMapper(compensation=CompensationStrategy.LUT, seed=42)
        w = np.random.default_rng(0).random((4, 4))
        result = mapper.map_weights(w)
        assert len(result.mappings[0].compensation_luts) > 0

    def test_no_compensation(self):
        mapper = MemristorMapper(compensation=CompensationStrategy.NONE, seed=42)
        w = np.random.default_rng(0).random((4, 4))
        result = mapper.map_weights(w)
        assert len(result.mappings[0].compensation_luts) == 0

    def test_all_technologies(self):
        for tech in MemristorTechnology:
            mapper = MemristorMapper(technology=tech, seed=42)
            w = np.random.default_rng(0).random((4, 4))
            result = mapper.map_weights(w)
            assert result.total_devices > 0

    def test_differential_topology(self):
        mapper = MemristorMapper(topology=CrossbarTopology.DIFFERENTIAL, seed=42)
        w = np.random.default_rng(0).random((4, 4))
        result = mapper.map_weights(w)
        assert result.mappings[0].crossbar.topology == CrossbarTopology.DIFFERENTIAL
        assert result.total_devices == 32


# ── MonteCarloSimulator Tests ────────────────────────────────────────


class TestMonteCarloSimulator:
    def test_simulate_mac(self):
        m = ConductanceModel(MemristorTechnology.GENERIC)
        sim = MonteCarloSimulator(m, num_trials=50, seed=42)
        w = np.random.default_rng(0).random((4, 4))
        inp = np.random.default_rng(1).random(4)
        report = sim.simulate_mac(w, inp)
        assert report.num_trials == 50
        assert report.mean_output_error >= 0

    def test_yield_bounded(self):
        m = ConductanceModel(MemristorTechnology.MYTHIC_AMP)
        sim = MonteCarloSimulator(m, num_trials=50, tolerance=0.5, seed=42)
        w = np.random.default_rng(0).random((4, 4))
        inp = np.ones(4) * 0.5
        report = sim.simulate_mac(w, inp)
        assert 0.0 <= report.yield_fraction <= 1.0

    def test_low_variability_high_yield(self):
        m = ConductanceModel(MemristorTechnology.MYTHIC_AMP)
        sim = MonteCarloSimulator(m, num_trials=100, tolerance=0.20, seed=42)
        w = np.ones((2, 2)) * 0.5
        inp = np.ones(2) * 0.5
        report = sim.simulate_mac(w, inp)
        assert report.yield_fraction > 0.5

    def test_error_histogram_shape(self):
        m = ConductanceModel(MemristorTechnology.GENERIC)
        sim = MonteCarloSimulator(m, num_trials=30, seed=42)
        w = np.random.default_rng(0).random((3, 3))
        inp = np.random.default_rng(1).random(3)
        report = sim.simulate_mac(w, inp)
        assert len(report.error_histogram) == 50

    def test_output_distribution_shape(self):
        m = ConductanceModel(MemristorTechnology.GENERIC)
        sim = MonteCarloSimulator(m, num_trials=20, seed=42)
        w = np.random.default_rng(0).random((3, 4))
        inp = np.random.default_rng(1).random(4)
        report = sim.simulate_mac(w, inp)
        assert report.output_distribution.shape == (3,)


# ── VerilogEmitter Tests ─────────────────────────────────────────────


class TestVerilogEmitter:
    def test_emit_crossbar_module(self):
        mapper = MemristorMapper(seed=42)
        w = np.random.default_rng(0).random((4, 4))
        result = mapper.map_weights(w)
        emitter = VerilogEmitter()
        sv = emitter.emit_crossbar(result.mappings[0])
        assert "module sc_memristor_crossbar" in sv
        assert "endmodule" in sv

    def test_emit_contains_spdx(self):
        mapper = MemristorMapper(seed=42)
        w = np.random.default_rng(0).random((4, 4))
        result = mapper.map_weights(w)
        emitter = VerilogEmitter()
        sv = emitter.emit_crossbar(result.mappings[0])
        assert "SPDX-License-Identifier" in sv

    def test_emit_weight_parameters(self):
        mapper = MemristorMapper(seed=42)
        w = np.random.default_rng(0).random((4, 4))
        result = mapper.map_weights(w)
        emitter = VerilogEmitter()
        sv = emitter.emit_crossbar(result.mappings[0])
        assert "W_0_0" in sv
        assert "W_3_3" in sv

    def test_emit_compensation_lut(self):
        mapper = MemristorMapper(compensation=CompensationStrategy.LUT, seed=42)
        w = np.random.default_rng(0).random((4, 4))
        result = mapper.map_weights(w)
        emitter = VerilogEmitter()
        sv = emitter.emit_crossbar(result.mappings[0])
        assert "comp_lut" in sv

    def test_emit_no_comp_when_none(self):
        mapper = MemristorMapper(compensation=CompensationStrategy.NONE, seed=42)
        w = np.random.default_rng(0).random((4, 4))
        result = mapper.map_weights(w)
        emitter = VerilogEmitter()
        sv = emitter.emit_crossbar(result.mappings[0])
        assert "No compensation LUT" in sv

    def test_emit_top_module(self):
        mapper = MemristorMapper(max_crossbar_size=4, seed=42)
        w = np.random.default_rng(0).random((8, 8))
        result = mapper.map_weights(w)
        emitter = VerilogEmitter()
        sv = emitter.emit_top(result)
        assert "module sc_memristor_array" in sv
        assert "tile_0" in sv
        assert "tile_1" in sv

    def test_custom_bit_width(self):
        mapper = MemristorMapper(seed=42)
        w = np.random.default_rng(0).random((2, 2))
        result = mapper.map_weights(w)
        emitter = VerilogEmitter(bit_width=32)
        sv = emitter.emit_crossbar(result.mappings[0])
        assert "[31:0]" in sv

    def test_emit_technology_in_header(self):
        mapper = MemristorMapper(technology=MemristorTechnology.PCM, seed=42)
        w = np.random.default_rng(0).random((4, 4))
        result = mapper.map_weights(w)
        emitter = VerilogEmitter()
        sv = emitter.emit_crossbar(result.mappings[0])
        assert "pcm" in sv


# ── Sneak-Path Model Tests ──────────────────────────────────────────


class TestSneakPathModel:
    def test_worst_case_sneak(self):
        sneak = SneakPathModel.worst_case_sneak(64, 64, 1e-6, 0.2)
        assert sneak > 0
        assert sneak == pytest.approx(126 * 1e-6 * 0.2)

    def test_larger_array_more_sneak(self):
        s1 = SneakPathModel.worst_case_sneak(32, 32, 1e-6)
        s2 = SneakPathModel.worst_case_sneak(128, 128, 1e-6)
        assert s2 > s1

    def test_signal_to_sneak_ratio(self):
        ratio = SneakPathModel.signal_to_sneak_ratio(100e-6, 1e-6, 64, 64)
        assert ratio > 0

    def test_1t1r_no_sneak_needed(self):
        ratio = SneakPathModel.signal_to_sneak_ratio(100e-6, 1e-6, 4, 4)
        assert ratio > 0


# ── IR-Drop Model Tests ─────────────────────────────────────────────


class TestIRDropModel:
    def test_corner_no_drop(self):
        ir = IRDropModel()
        assert ir.voltage_drop(0, 0) == 0.0

    def test_drop_increases_with_position(self):
        ir = IRDropModel()
        d1 = ir.voltage_drop(10, 10)
        d2 = ir.voltage_drop(50, 50)
        assert d2 > d1

    def test_effective_conductance_reduced(self):
        ir = IRDropModel(r_wire_per_cell=5.0)
        g_nom = 50e-6
        g_eff = ir.effective_conductance(g_nom, 100, 100, v_read=0.2)
        assert g_eff < g_nom

    def test_zero_drop_at_corner(self):
        ir = IRDropModel()
        g_nom = 50e-6
        g_eff = ir.effective_conductance(g_nom, 0, 0)
        assert g_eff == g_nom


# ── Stuck Fault Model Tests ─────────────────────────────────────────


class TestStuckFaultMap:
    def test_generate_faults(self):
        fm = StuckFaultMap.generate(100, 100, fault_rate=0.01, seed=42)
        assert fm.num_faults > 0
        assert fm.fault_rate > 0

    def test_is_stuck(self):
        fm = StuckFaultMap(10, 10, stuck_on=[(0, 0)], stuck_off=[(1, 1)])
        assert fm.is_stuck(0, 0) == "on"
        assert fm.is_stuck(1, 1) == "off"
        assert fm.is_stuck(5, 5) is None

    def test_zero_rate_no_faults(self):
        fm = StuckFaultMap.generate(10, 10, fault_rate=0.0, seed=42)
        assert fm.num_faults == 0

    def test_fault_rate_property(self):
        fm = StuckFaultMap(10, 10, stuck_on=[(0, 0)], stuck_off=[(1, 1)])
        assert fm.fault_rate == pytest.approx(0.02)


# ── Aging / Drift Tests ─────────────────────────────────────────────


class TestAgingSimulator:
    def test_drift_reduces_conductance(self):
        m = ConductanceModel(MemristorTechnology.GENERIC)
        g0 = 50e-6
        g_drifted = m.drift(g0, elapsed_s=3.15e7)
        assert g_drifted < g0

    def test_no_drift_at_t0(self):
        m = ConductanceModel(MemristorTechnology.GENERIC)
        g0 = 50e-6
        assert m.drift(g0, elapsed_s=0.5) == g0

    def test_aging_simulator(self):
        m = ConductanceModel(MemristorTechnology.GENERIC)
        inj = VariabilityInjector(m, seed=42)
        w = np.ones((4, 4)) * 0.5
        _, g = inj.inject_full(w)
        sim = AgingSimulator(m)
        drifted, report = sim.simulate(g, elapsed_s=3.15e7)
        assert report.mean_drift_fraction > 0
        assert np.all(drifted <= g)

    def test_short_time_no_shift(self):
        m = ConductanceModel(MemristorTechnology.GENERIC)
        g = np.full((2, 2), 50e-6)
        sim = AgingSimulator(m)
        _, report = sim.simulate(g, elapsed_s=0.5)
        assert report.mean_drift_fraction == 0.0


# ── Thermal Shift Tests ─────────────────────────────────────────────


class TestThermalShift:
    def test_higher_temp_shifts(self):
        m = ConductanceModel(MemristorTechnology.GENERIC)
        g0 = 50e-6
        g_hot = m.thermal_shift(g0, temp_c=85.0)
        assert g_hot != g0

    def test_ref_temp_no_shift(self):
        m = ConductanceModel(MemristorTechnology.GENERIC)
        g0 = 50e-6
        g_ref = m.thermal_shift(g0, temp_c=25.0)
        assert g_ref == pytest.approx(g0)


# ── SC Absorb Encoder Tests ─────────────────────────────────────────


class TestSCAbsorbEncoder:
    def test_adjusted_thresholds_shape(self):
        m = ConductanceModel(MemristorTechnology.GENERIC)
        inj = VariabilityInjector(m, seed=42)
        w = np.random.default_rng(0).random((4, 4))
        _, g = inj.inject_full(w)
        thresholds = SCAbsorbEncoder.compute_adjusted_thresholds(w, g, m)
        assert thresholds.shape == (4, 4)

    def test_ideal_gives_256(self):
        m = ConductanceModel(MemristorTechnology.GENERIC)
        w = np.ones((2, 2)) * 0.5
        g_ideal = np.array([[m.target_conductance(int(round(0.5 * (m.num_levels - 1))))] * 2] * 2)
        thresholds = SCAbsorbEncoder.compute_adjusted_thresholds(w, g_ideal, m)
        assert np.all(thresholds == 256)

    def test_deviated_compensates(self):
        m = ConductanceModel(MemristorTechnology.GENERIC)
        w = np.ones((2, 2)) * 0.5
        g_deviated = (
            np.ones((2, 2)) * m.target_conductance(int(round(0.5 * (m.num_levels - 1)))) * 0.8
        )
        thresholds = SCAbsorbEncoder.compute_adjusted_thresholds(w, g_deviated, m)
        assert np.all(thresholds > 256)


# ── Write-Verify Protocol Tests ─────────────────────────────────────


class TestWriteVerifyProtocol:
    def test_converges(self):
        m = ConductanceModel(MemristorTechnology.GENERIC)
        wv = WriteVerifyProtocol(m, max_iterations=20, tolerance=0.05, seed=42)
        result = wv.program_cell(8)
        assert result.iterations > 0
        assert result.target_level == 8

    def test_low_tolerance_may_need_more_iterations(self):
        m = ConductanceModel(MemristorTechnology.GENERIC)
        wv1 = WriteVerifyProtocol(m, max_iterations=20, tolerance=0.10, seed=42)
        wv2 = WriteVerifyProtocol(m, max_iterations=20, tolerance=0.001, seed=42)
        r1 = wv1.program_cell(8)
        r2 = wv2.program_cell(8)
        assert r1.iterations <= r2.iterations or not r2.converged


# ── Power Estimation Tests ───────────────────────────────────────────


class TestCrossbarEstimator:
    def test_estimate_standard(self):
        xbar = CrossbarArray(64, 64, technology=MemristorTechnology.RERAM_HFOX)
        est = CrossbarEstimator.estimate(xbar)
        assert est.read_power_uw > 0
        assert est.write_power_uw > est.read_power_uw
        assert est.area_um2 > 0

    def test_2d_lower_area(self):
        xbar_hfox = CrossbarArray(64, 64, technology=MemristorTechnology.RERAM_HFOX)
        xbar_2d = CrossbarArray(64, 64, technology=MemristorTechnology.RERAM_2D)
        e1 = CrossbarEstimator.estimate(xbar_hfox)
        e2 = CrossbarEstimator.estimate(xbar_2d)
        assert e2.area_um2 < e1.area_um2

    def test_all_technologies(self):
        for tech in MemristorTechnology:
            xbar = CrossbarArray(16, 16, technology=tech)
            est = CrossbarEstimator.estimate(xbar)
            assert est.read_latency_ns > 0
            assert est.write_latency_ns > 0


def test_signal_to_sneak_ratio_is_infinite_without_sneak_current() -> None:
    # A zero off-conductance gives zero worst-case sneak current, so the
    # signal-to-sneak ratio is infinite rather than dividing by zero.
    assert SneakPathModel.signal_to_sneak_ratio(g_on=1e-3, g_off=0.0, rows=8, cols=8) == float(
        "inf"
    )
