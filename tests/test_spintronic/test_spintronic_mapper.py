# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Spintronic Mapper Tests

import numpy as np
import pytest

from sc_neurocore.spintronic.spintronic_mapper import (
    AgingModel,
    DefectMap,
    MLCConfig,
    MaterialParams,
    MuMax3OutputParser,
    MuMax3Result,
    MuMax3ScriptGenerator,
    RacetrackShiftRegister,
    RadiationModel,
    SkyrmionHallCorrector,
    SpintronicArray,
    SpintronicCell,
    SpintronicDeviceConfig,
    SpintronicMapper,
    SpintronicTech,
    SpintronicVerilogGenerator,
    VariabilityModel,
    retention_failure_probability,
    switching_current_vs_temperature,
    switching_time_vs_temperature,
    write_verify,
)


# ── MaterialParams Tests ─────────────────────────────────────────────


class TestMaterialParams:
    def test_cofeb_mgo(self):
        m = MaterialParams.cofeb_mgo()
        assert m.saturation_magnetisation_a_m > 0
        assert m.damping_alpha > 0

    def test_pt_co(self):
        m = MaterialParams.pt_co_multilayer()
        assert m.dmi_strength_j_m2 > 0  # skyrmion host requires DMI

    def test_w_cofeb(self):
        m = MaterialParams.w_cofeb()
        assert m.saturation_magnetisation_a_m > 0


# ── SpintronicDeviceConfig Tests ─────────────────────────────────────


class TestSpintronicDeviceConfig:
    def test_all_techs(self):
        for tech in SpintronicTech:
            cfg = SpintronicDeviceConfig.from_tech(tech)
            assert cfg.width_nm > 0
            assert cfg.switching_current_ua > 0

    def test_area(self):
        cfg = SpintronicDeviceConfig.from_tech(SpintronicTech.SOT_MRAM)
        assert cfg.area_nm2 == cfg.width_nm * cfg.length_nm

    def test_switching_energy(self):
        cfg = SpintronicDeviceConfig.from_tech(SpintronicTech.SOT_MRAM)
        assert cfg.switching_energy_fj > 0

    def test_switching_energy_uses_device_write_resistance(self):
        low_r = SpintronicDeviceConfig(
            switching_current_ua=40.0,
            switching_time_ns=2.0,
            write_resistance_ohm=2_000.0,
        )
        high_r = SpintronicDeviceConfig(
            switching_current_ua=40.0,
            switching_time_ns=2.0,
            write_resistance_ohm=8_000.0,
        )
        assert high_r.switching_energy_fj == 4.0 * low_r.switching_energy_fj

    def test_skyrmion_has_dmi(self):
        cfg = SpintronicDeviceConfig.from_tech(SpintronicTech.SKYRMION)
        assert cfg.material.dmi_strength_j_m2 > 0

    def test_sot_faster_than_stt(self):
        sot = SpintronicDeviceConfig.from_tech(SpintronicTech.SOT_MRAM)
        stt = SpintronicDeviceConfig.from_tech(SpintronicTech.STT_MTJ)
        assert sot.switching_time_ns < stt.switching_time_ns

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"width_nm": 0.0}, "width_nm must be positive"),
            ({"length_nm": 0.0}, "length_nm must be positive"),
            ({"thickness_nm": 0.0}, "thickness_nm must be positive"),
            ({"switching_current_ua": 0.0}, "switching_current_ua must be positive"),
            ({"switching_time_ns": 0.0}, "switching_time_ns must be positive"),
            ({"write_resistance_ohm": 0.0}, "write_resistance_ohm must be positive"),
            ({"parallel_resistance_ohm": 0.0}, "parallel_resistance_ohm must be positive"),
            ({"tmr_ratio": -0.1}, "tmr_ratio must be non-negative"),
        ],
    )
    def test_rejects_each_invalid_field(self, kwargs: dict[str, float], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            SpintronicDeviceConfig(**kwargs)


# ── VariabilityModel Tests ───────────────────────────────────────────


class TestVariabilityModel:
    def test_apply(self):
        rng = np.random.default_rng(42)
        base = SpintronicDeviceConfig.from_tech(SpintronicTech.SOT_MRAM)
        var = VariabilityModel()
        varied = var.apply(base, rng)
        assert varied.width_nm != base.width_nm or varied.length_nm != base.length_nm

    def test_apply_clamps(self):
        rng = np.random.default_rng(42)
        base = SpintronicDeviceConfig.from_tech(SpintronicTech.SOT_MRAM)
        var = VariabilityModel(width_sigma_pct=500)
        varied = var.apply(base, rng)
        assert varied.width_nm >= 10.0
        assert varied.material.damping_alpha >= 0.001

    def test_zero_variability(self):
        rng = np.random.default_rng(42)
        var = VariabilityModel(
            width_sigma_pct=0,
            length_sigma_pct=0,
            ku_sigma_pct=0,
            dmi_sigma_pct=0,
            damping_sigma_pct=0,
            ms_sigma_pct=0,
        )
        base = SpintronicDeviceConfig.from_tech(SpintronicTech.SOT_MRAM)
        varied = var.apply(base, rng)
        assert abs(varied.width_nm - base.width_nm) < 1e-6


# ── SpintronicArray Tests ────────────────────────────────────────────


class TestSpintronicArray:
    def test_creation(self):
        arr = SpintronicArray(4, 8)
        assert arr.total_cells == 32

    def test_total_area(self):
        arr = SpintronicArray(4, 4)
        assert arr.total_area_um2 > 0

    def test_program_and_read(self):
        arr = SpintronicArray(
            2,
            3,
            variability=VariabilityModel(
                width_sigma_pct=0,
                length_sigma_pct=0,
                ku_sigma_pct=0,
                dmi_sigma_pct=0,
                damping_sigma_pct=0,
                ms_sigma_pct=0,
            ),
        )
        w = np.array([[100, 200, 50], [250, 10, 180]], dtype=np.int32)
        arr.program_weights(w)
        rb = arr.read_weights()
        np.testing.assert_array_equal(rb, w)

    def test_state_from_weight(self):
        arr = SpintronicArray(
            1,
            2,
            variability=VariabilityModel(
                width_sigma_pct=0,
                length_sigma_pct=0,
                ku_sigma_pct=0,
                dmi_sigma_pct=0,
                damping_sigma_pct=0,
                ms_sigma_pct=0,
            ),
        )
        w = np.array([[50, 200]], dtype=np.int32)
        arr.program_weights(w)
        assert arr.cells[0][0].state == 0  # w=50 < 128 → P
        assert arr.cells[0][1].state == 1  # w=200 > 128 → AP


# ── SpintronicCell Tests ─────────────────────────────────────────────


class TestSpintronicCell:
    def test_resistance_p(self):
        dev = SpintronicDeviceConfig.from_tech(SpintronicTech.SOT_MRAM)
        cell = SpintronicCell(0, 0, dev, state=0)
        r_p = cell.resistance_ohm
        assert r_p == dev.parallel_resistance_ohm

    def test_resistance_ap(self):
        dev = SpintronicDeviceConfig.from_tech(SpintronicTech.SOT_MRAM)
        cell = SpintronicCell(0, 0, dev, state=1)
        r_ap = cell.resistance_ohm
        assert r_ap > 5000.0

    def test_tmr_ratio(self):
        dev = SpintronicDeviceConfig.from_tech(SpintronicTech.SOT_MRAM)
        p = SpintronicCell(0, 0, dev, state=0)
        ap = SpintronicCell(0, 0, dev, state=1)
        ratio = (ap.resistance_ohm - p.resistance_ohm) / p.resistance_ohm
        assert abs(ratio - dev.tmr_ratio) < 0.01

    def test_resistance_uses_device_parallel_resistance(self):
        dev = SpintronicDeviceConfig(
            parallel_resistance_ohm=7_500.0,
            tmr_ratio=2.0,
        )
        p = SpintronicCell(0, 0, dev, state=0)
        ap = SpintronicCell(0, 0, dev, state=1)
        assert p.resistance_ohm == 7_500.0
        assert ap.resistance_ohm == 22_500.0


# ── SpintronicMapper Tests ───────────────────────────────────────────


class TestSpintronicMapper:
    def test_map_network(self):
        mapper = SpintronicMapper()
        w = np.random.default_rng(42).integers(0, 256, (8, 16), dtype=np.int32)
        arr, result = mapper.map_network(w)
        assert result.array_rows == 8
        assert result.array_cols == 16
        assert result.total_energy_fj > 0

    def test_all_techs(self):
        w = np.ones((4, 4), dtype=np.int32) * 128
        for tech in SpintronicTech:
            mapper = SpintronicMapper(tech=tech)
            arr, result = mapper.map_network(w)
            assert result.tech == tech
            assert result.total_area_um2 > 0

    def test_monte_carlo_yield(self):
        mapper = SpintronicMapper()
        w = np.ones((4, 4), dtype=np.int32) * 128
        yld = mapper.monte_carlo_yield(w, n_trials=50, tolerance_q88=128)
        assert 0.0 <= yld <= 1.0

    def test_yield_high_tolerance(self):
        mapper = SpintronicMapper()
        w = np.ones((4, 4), dtype=np.int32) * 128
        yld = mapper.monte_carlo_yield(w, n_trials=20, tolerance_q88=256)
        assert yld == 1.0  # very high tolerance → 100% yield


# ── MuMax3 Tests ─────────────────────────────────────────────────────


class TestMuMax3ScriptGenerator:
    def test_switching_script(self):
        dev = SpintronicDeviceConfig.from_tech(SpintronicTech.SOT_MRAM)
        script = MuMax3ScriptGenerator.generate_switching(dev)
        assert "Msat" in script
        assert "Aex" in script
        assert "Run(" in script

    def test_skyrmion_script(self):
        dev = SpintronicDeviceConfig.from_tech(SpintronicTech.SKYRMION)
        script = MuMax3ScriptGenerator.generate_skyrmion(dev)
        assert "Skyrmion" in script
        assert "Relax" in script
        assert "Dind" in script


# ── Verilog Generator Tests ──────────────────────────────────────────


class TestSpintronicVerilogGenerator:
    def test_generate(self):
        v = SpintronicVerilogGenerator.generate(
            "sc_spin_array",
            8,
            16,
            SpintronicTech.SOT_MRAM,
        )
        assert "module sc_spin_array" in v
        assert "ROWS = 8" in v
        assert "COLS = 16" in v

    def test_has_programming_interface(self):
        v = SpintronicVerilogGenerator.generate(
            "test",
            4,
            4,
            SpintronicTech.SKYRMION,
        )
        assert "prog_en" in v
        assert "prog_weight" in v


# ── Thermal Stability Tests ──────────────────────────────────────────


class TestThermalStability:
    def test_thermal_stability_positive(self):
        for tech in SpintronicTech:
            cfg = SpintronicDeviceConfig.from_tech(tech)
            assert cfg.thermal_stability > 0

    def test_larger_device_more_stable(self):
        small = SpintronicDeviceConfig.from_tech(SpintronicTech.SOT_MRAM)
        import copy

        large = copy.deepcopy(small)
        large.width_nm *= 2
        large.length_nm *= 2
        assert large.thermal_stability > small.thermal_stability

    def test_sot_adequate_retention(self):
        cfg = SpintronicDeviceConfig.from_tech(SpintronicTech.SOT_MRAM)
        # Δ > 1 is basic sanity; real devices need > 40
        assert cfg.thermal_stability > 1.0


# ── Read Disturb Tests ───────────────────────────────────────────────


class TestReadDisturb:
    def test_read_disturb_low(self):
        cfg = SpintronicDeviceConfig.from_tech(SpintronicTech.SOT_MRAM)
        assert cfg.read_disturb_probability < 1.0

    def test_read_disturb_nonnegative(self):
        for tech in SpintronicTech:
            cfg = SpintronicDeviceConfig.from_tech(tech)
            assert cfg.read_disturb_probability >= 0.0


# ── Endurance Tests ──────────────────────────────────────────────────


class TestEndurance:
    def test_endurance_positive(self):
        for tech in SpintronicTech:
            cfg = SpintronicDeviceConfig.from_tech(tech)
            assert cfg.endurance_cycles > 0

    def test_sot_higher_than_stt(self):
        sot = SpintronicDeviceConfig.from_tech(SpintronicTech.SOT_MRAM)
        stt = SpintronicDeviceConfig.from_tech(SpintronicTech.STT_MTJ)
        assert sot.endurance_cycles >= stt.endurance_cycles


# ── Power Breakdown Tests ────────────────────────────────────────────


class TestPowerBreakdown:
    def test_power_breakdown_keys(self):
        arr = SpintronicArray(2, 2)
        pb = arr.power_breakdown(bitstream_length=128)
        assert "switching_fj" in pb
        assert "leakage_fj" in pb
        assert "total_fj" in pb

    def test_total_equals_sum(self):
        arr = SpintronicArray(4, 4)
        pb = arr.power_breakdown(256)
        assert abs(pb["total_fj"] - pb["switching_fj"] - pb["leakage_fj"]) < 1e-6

    def test_longer_bitstream_more_energy(self):
        arr = SpintronicArray(4, 4)
        pb_short = arr.power_breakdown(128)
        pb_long = arr.power_breakdown(512)
        assert pb_long["total_fj"] > pb_short["total_fj"]

    def test_energy_positive(self):
        arr = SpintronicArray(2, 2)
        pb = arr.power_breakdown(256)
        assert pb["switching_fj"] > 0
        assert pb["leakage_fj"] > 0


# ── Racetrack Shift Register Tests (Gap 1) ────────────────────────────


class TestRacetrackShiftRegister:
    def test_load_and_shift(self):
        rt = RacetrackShiftRegister(n_positions=8)
        rt.load(np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.int8))
        rt.shift_right()
        assert rt.bits[0] == 0  # shifted in zero
        assert rt.bits[1] == 1  # original bit[0]

    def test_shift_left(self):
        rt = RacetrackShiftRegister(n_positions=4)
        rt.load(np.array([1, 0, 1, 0], dtype=np.int8))
        rt.shift_left()
        assert rt.bits[-1] == 0
        assert rt.bits[0] == 0  # original bit[1]

    def test_shift_energy(self):
        rt = RacetrackShiftRegister(n_positions=8)
        assert rt.shift_energy_fj > 0

    def test_shift_right_injects_error_under_rng(self):
        # With a certain shift-error rate, the rng-driven bit flip path is taken.
        rt = RacetrackShiftRegister(n_positions=8, shift_error_rate=1.0)
        rt.load(np.zeros(8, dtype=np.int8))
        rt.shift_right(rng=np.random.default_rng(0))
        assert int(rt.bits.sum()) == 1  # the single injected flip

    def test_shift_left_injects_error_under_rng(self):
        rt = RacetrackShiftRegister(n_positions=8, shift_error_rate=1.0)
        rt.load(np.zeros(8, dtype=np.int8))
        rt.shift_left(rng=np.random.default_rng(0))
        assert int(rt.bits.sum()) == 1


# ── Skyrmion Hall Angle Tests (Gap 2) ─────────────────────────────────


class TestSkyrmionHall:
    def test_hall_angle(self):
        shc = SkyrmionHallCorrector()
        assert shc.hall_angle_deg > 0

    def test_corrected_position(self):
        shc = SkyrmionHallCorrector()
        x, y = shc.corrected_position(100.0, 50.0)
        assert x == 100.0
        assert abs(y) <= 25.0  # clamped to track width

    def test_needs_confinement(self):
        shc = SkyrmionHallCorrector()
        assert isinstance(shc.needs_confinement, bool)


# ── Temperature-Dependent Switching Tests (Gap 3) ─────────────────────


class TestTempSwitching:
    def test_current_decreases_with_temp(self):
        ic_cold = switching_current_vs_temperature(50.0, 40.0, 200.0)
        ic_hot = switching_current_vs_temperature(50.0, 40.0, 400.0)
        assert ic_cold > ic_hot

    def test_time_increases_with_temp(self):
        t_cold = switching_time_vs_temperature(1.0, 200.0)
        t_hot = switching_time_vs_temperature(1.0, 400.0)
        assert t_hot > t_cold

    def test_current_degenerate_parameters_return_baseline(self):
        # A non-positive stability barrier leaves the model undefined, so the
        # baseline critical current is returned unchanged.
        assert switching_current_vs_temperature(50.0, 0.0, 300.0) == 50.0


# ── Retention Failure Tests (Gap 4) ───────────────────────────────────


class TestRetentionFailure:
    def test_high_stability_no_fail(self):
        assert retention_failure_probability(101.0, 3.15e8) == 0.0  # Δ>100 → 0

    def test_low_stability_fails(self):
        p = retention_failure_probability(10.0, 1.0)
        assert p > 0.0


# ── MLC Tests (Gap 5) ─────────────────────────────────────────────────


class TestMLCConfig:
    def test_levels(self):
        mlc = MLCConfig(bits_per_cell=2)
        assert mlc.levels == 4

    def test_quantize(self):
        mlc = MLCConfig(bits_per_cell=2)
        assert mlc.quantize_weight(0.0) == 0
        assert mlc.quantize_weight(1.0) == 3

    def test_dequantize(self):
        mlc = MLCConfig(bits_per_cell=2)
        assert mlc.dequantize(0) == 0.0
        assert abs(mlc.dequantize(3) - 1.0) < 0.01

    def test_density(self):
        assert MLCConfig(bits_per_cell=3).density_improvement == 3.0

    def test_resistance_margins_span_parallel_to_antiparallel(self):
        margins = MLCConfig(bits_per_cell=2).resistance_margins
        assert len(margins) == 4
        assert margins[0] == 5000.0
        assert margins[-1] == 12500.0


# ── Write-Verify Tests (Gap 6) ────────────────────────────────────────


class TestWriteVerify:
    def test_success(self):
        dev = SpintronicDeviceConfig.from_tech(SpintronicTech.SOT_MRAM)
        cell = SpintronicCell(0, 0, dev)
        result = write_verify(cell, 200)
        assert result.success
        assert result.error <= 4

    def test_with_noise(self):
        dev = SpintronicDeviceConfig.from_tech(SpintronicTech.SOT_MRAM)
        cell = SpintronicCell(0, 0, dev)
        rng = np.random.default_rng(42)
        result = write_verify(cell, 200, rng=rng)
        assert result.attempts >= 1

    def test_exhausts_attempts_when_noise_never_settles(self):
        # A noise source that always overshoots the tolerance forces every
        # attempt to miss, so the loop reports failure after max_attempts.
        class _AlwaysFarNoise:
            def normal(self, _mean: float, _std: float) -> float:
                return 100.0

        dev = SpintronicDeviceConfig.from_tech(SpintronicTech.SOT_MRAM)
        cell = SpintronicCell(0, 0, dev)
        result = write_verify(cell, 200, max_attempts=3, rng=_AlwaysFarNoise())
        assert result.success is False
        assert result.attempts == 3


# ── Aging Model Tests (Gap 7) ─────────────────────────────────────────


class TestAgingModel:
    def test_no_degradation_initially(self):
        am = AgingModel()
        assert am.tmr_degradation(1.5, 10**12) == 1.5

    def test_degradation_with_cycles(self):
        am = AgingModel(cycles_written=10**12)
        tmr = am.tmr_degradation(1.5, 10**12)
        assert tmr < 1.5

    def test_write_increments(self):
        am = AgingModel()
        am.write(100)
        assert am.cycles_written == 100

    def test_tmr_degradation_zero_endurance_is_identity(self):
        assert AgingModel(cycles_written=10).tmr_degradation(1.5, 0) == 1.5

    def test_stability_degradation_zero_endurance_is_identity(self):
        assert AgingModel(cycles_written=10).stability_degradation(2.0, 0) == 2.0

    def test_stability_degradation_with_cycles(self):
        degraded = AgingModel(cycles_written=10**12).stability_degradation(2.0, 10**12)
        assert degraded < 2.0

    def test_is_worn_out_flag(self):
        am = AgingModel(cycles_written=10)
        assert isinstance(am.is_worn_out, bool)


# ── Radiation Model Tests (Gap 8) ─────────────────────────────────────


class TestRadiationModel:
    def test_is_rad_hard(self):
        rm = RadiationModel()
        assert rm.is_rad_hard

    def test_seu_rate(self):
        rm = RadiationModel()
        rate = rm.seu_rate(1e4, 1000)  # LEO flux
        assert rate > 0

    def test_tid_degradation(self):
        rm = RadiationModel()
        assert rm.tid_degradation(0.0) == 1.0
        assert rm.tid_degradation(1000.0) == 0.5


# ── Defect Map Tests (Gap 9) ───────────────────────────────────────────


class TestDefectMap:
    def test_add_and_count(self):
        dm = DefectMap()
        dm.add_defect(0, 3, "stuck_p")
        assert dm.defect_count == 1
        assert dm.is_defective(0, 3)

    def test_remap(self):
        dm = DefectMap()
        dm.add_defect(0, 3, "stuck_p")
        dm.add_remap((0, 3), (7, 0))
        assert dm.effective_address(0, 3) == (7, 0)

    def test_defect_rate(self):
        dm = DefectMap()
        dm.add_defect(0, 0, "open")
        assert dm.defect_rate(100) == 0.01

    def test_defect_rate_zero_cells(self):
        dm = DefectMap()
        dm.add_defect(0, 0, "open")
        assert dm.defect_rate(0) == 0.0


# ── MuMax3 Parser Tests (Gap 10) ──────────────────────────────────────


class TestMuMax3Parser:
    def test_parse_table(self):
        table = "# t mx my mz\n5e-9\t0.01\t0.02\t-0.99"
        result = MuMax3OutputParser.parse_table(table)
        assert result.switched is True
        assert result.final_mz < 0

    def test_successful_switch(self):
        r = MuMax3Result(0.01, 0.02, -0.99, True)
        assert MuMax3OutputParser.is_switching_successful(r)

    def test_failed_switch(self):
        r = MuMax3Result(0.01, 0.02, 0.99, False)
        assert not MuMax3OutputParser.is_switching_successful(r)

    def test_empty_input(self):
        result = MuMax3OutputParser.parse_table("")
        assert result.final_mz == 0.0

    def test_parse_table_whitespace_separated_row(self):
        # A row that is space- rather than tab-separated falls back to a generic
        # whitespace split and still parses.
        table = "# t mx my mz\n5e-9 0.01 0.02 -0.99"
        result = MuMax3OutputParser.parse_table(table)
        assert result.switched is True
        assert result.final_mz < 0

    def test_parse_table_non_numeric_row_returns_default(self):
        # A malformed row that cannot be parsed as floats yields a default
        # result rather than raising.
        result = MuMax3OutputParser.parse_table("# header\nnot a number row")
        assert result.final_mz == 0.0
        assert result.switched is False
