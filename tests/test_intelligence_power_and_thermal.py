# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore

import unittest

import pytest

from sc_neurocore.compiler.intelligence import (
    configure_approximation,
    generate_dvfs_controller,
    model_energy_harvest,
)


class TestApproximation(unittest.TestCase):
    def test_basic(self):
        r = configure_approximation({"v": "-(v)/tau + I"})
        self.assertGreater(r.total_energy_savings_pct, 0)
        self.assertIn("v", r.populations)
        self.assertIn("bits_reduced", r.populations["v"])

    def test_error_bound(self):
        r = configure_approximation(
            {"v": "a", "u": "b"},
            max_error_pct=2.0,
        )
        self.assertLessEqual(r.max_output_error_pct, 3.1)

    def test_multi_var(self):
        r = configure_approximation({"v": "a", "u": "b", "w": "c"})
        self.assertEqual(len(r.populations), 3)


class TestEnergyHarvest(unittest.TestCase):
    def test_solar_outdoor(self):
        r = model_energy_harvest(
            100.0, harvester_type="solar", environment="outdoor", harvester_area_cm2=1.0
        )
        self.assertTrue(r.energy_positive)
        self.assertGreater(r.margin_pct, 0)

    def test_rf_indoor_insufficient(self):
        r = model_energy_harvest(100.0, harvester_type="rf", environment="indoor")
        self.assertFalse(r.energy_positive)
        self.assertLess(r.recommended_duty_cycle, 1.0)


class TestDVFS(unittest.TestCase):
    def test_default(self):
        v = generate_dvfs_controller("sc_lif")
        self.assertIn("module sc_lif_dvfs_ctrl", v)
        self.assertIn("spike_rate", v)
        self.assertIn("OP_0", v)
        self.assertIn("endmodule", v)

    def test_custom_points(self):
        v = generate_dvfs_controller(
            "sc_hh",
            operating_points=[
                {"voltage_mv": 600, "freq_mhz": 50},
                {"voltage_mv": 1200, "freq_mhz": 800},
            ],
        )
        self.assertIn("50", v)
        self.assertIn("800", v)


class TestThermalAnalysis:
    """Tests for thermal estimation and derating."""

    def test_basic_thermal(self):
        """Basic thermal analysis returns valid fields."""
        from sc_neurocore.compiler.intelligence import thermal_analysis

        t = thermal_analysis(100.0, 500.0)
        assert t.junction_temp_c > 25.0
        assert t.derated_freq_mhz > 0
        assert t.thermal_safe
        assert t.hotspot_risk in ("none", "low", "medium", "high")

    def test_low_power_safe(self):
        """Low power design is thermally safe."""
        from sc_neurocore.compiler.intelligence import thermal_analysis

        t = thermal_analysis(0.1, 100.0)
        assert t.thermal_safe
        assert t.delta_t_c < 1.0

    def test_high_power_derating(self):
        """High power causes frequency derating."""
        from sc_neurocore.compiler.intelligence import thermal_analysis

        t = thermal_analysis(10000.0, 500.0)  # 10W
        assert t.junction_temp_c > 85.0
        assert t.derated_freq_mhz < 500.0

    def test_extreme_power_unsafe(self):
        """Extreme power exceeds junction limit."""
        from sc_neurocore.compiler.intelligence import thermal_analysis

        t = thermal_analysis(50000.0, 500.0)  # 50W
        assert not t.thermal_safe

    def test_dsp_hotspot(self):
        """Many DSPs in one column → high hotspot risk."""
        from sc_neurocore.compiler.intelligence import thermal_analysis

        t = thermal_analysis(100.0, 500.0, mul_count=30, dsp_columns=1)
        assert t.hotspot_risk == "high"

    def test_hotspot_concentration_derates_frequency(self):
        """Concentrated DSP hotspots should affect timing, not only labels."""
        from sc_neurocore.compiler.intelligence import thermal_analysis

        spread = thermal_analysis(100.0, 500.0, mul_count=30, dsp_columns=10)
        concentrated = thermal_analysis(100.0, 500.0, mul_count=30, dsp_columns=1)

        assert concentrated.hotspot_risk == "high"
        assert concentrated.derated_freq_mhz < spread.derated_freq_mhz

    def test_dsp_hotspot_adds_local_junction_rise(self):
        """DSP hotspot power should increase junction temperature, not only risk labels."""
        from sc_neurocore.compiler.intelligence import thermal_analysis

        spread = thermal_analysis(
            1000.0,
            500.0,
            mul_count=32,
            dsp_columns=8,
            dsp_power_mw=320.0,
            theta_spreading=12.0,
        )
        concentrated = thermal_analysis(
            1000.0,
            500.0,
            mul_count=32,
            dsp_columns=1,
            dsp_power_mw=320.0,
            theta_spreading=12.0,
        )

        assert concentrated.hotspot_delta_t_c > spread.hotspot_delta_t_c
        assert concentrated.junction_temp_c > spread.junction_temp_c

    def test_dsp_spread(self):
        """DSPs spread across columns → lower risk."""
        from sc_neurocore.compiler.intelligence import thermal_analysis

        t = thermal_analysis(100.0, 500.0, mul_count=30, dsp_columns=10)
        assert t.hotspot_risk in ("none", "low")

    def test_small_process_more_derating(self):
        """7nm process derates more than 28nm."""
        from sc_neurocore.compiler.intelligence import thermal_analysis

        t7 = thermal_analysis(5000.0, 500.0, process_nm=7)
        t28 = thermal_analysis(5000.0, 500.0, process_nm=28)
        assert t7.derated_freq_mhz < t28.derated_freq_mhz

    def test_rejects_invalid_physical_inputs(self):
        """Thermal analysis must reject non-physical parameters."""
        from sc_neurocore.compiler.intelligence import thermal_analysis

        invalid_cases = [
            ({"estimated_power_mw": -1.0, "target_freq_mhz": 500.0}, "estimated_power_mw"),
            ({"estimated_power_mw": 1.0, "target_freq_mhz": 0.0}, "target_freq_mhz"),
            ({"estimated_power_mw": 1.0, "target_freq_mhz": 500.0, "theta_ja": 0.0}, "theta_ja"),
            (
                {"estimated_power_mw": 1.0, "target_freq_mhz": 500.0, "process_nm": 0},
                "process_nm",
            ),
            (
                {"estimated_power_mw": 1.0, "target_freq_mhz": 500.0, "mul_count": -1},
                "mul_count",
            ),
            (
                {"estimated_power_mw": 1.0, "target_freq_mhz": 500.0, "dsp_columns": 0},
                "dsp_columns",
            ),
        ]
        for kwargs, message in invalid_cases:
            with pytest.raises(ValueError, match=message):
                thermal_analysis(**kwargs)


class TestThermalConstraints:
    """Tests for thermal constraint generation."""

    def test_basic_constraints(self):
        """Thermal constraints include derated clock."""
        from sc_neurocore.compiler.intelligence import (
            thermal_analysis,
            generate_thermal_constraints,
        )

        t = thermal_analysis(100.0, 500.0)
        xdc = generate_thermal_constraints("sc_lif", t)
        assert "create_clock" in xdc
        assert "Derated frequency" in xdc

    def test_hotspot_constraints(self):
        """High hotspot risk adds DSP spreading."""
        from sc_neurocore.compiler.intelligence import (
            thermal_analysis,
            generate_thermal_constraints,
        )

        t = thermal_analysis(100.0, 500.0, mul_count=25, dsp_columns=1)
        xdc = generate_thermal_constraints("sc_hh", t)
        assert "DSP spreading" in xdc

    def test_unsafe_warning(self):
        """Unsafe temperature adds warning."""
        from sc_neurocore.compiler.intelligence import (
            thermal_analysis,
            generate_thermal_constraints,
        )

        t = thermal_analysis(50000.0, 500.0)
        xdc = generate_thermal_constraints("sc_lif", t)
        assert "WARNING" in xdc


class TestEnergyScheduler:
    """Energy-aware neuron scheduling."""

    def test_basic_schedule(self):
        from sc_neurocore.compiler.intelligence import (
            generate_energy_schedule,
        )

        s = generate_energy_schedule(1000)
        assert s.total_neurons == 1000
        assert s.neurons_per_epoch <= 1000
        assert s.duty_cycle > 0

    def test_energy_limited(self):
        from sc_neurocore.compiler.intelligence import (
            generate_energy_schedule,
        )

        s = generate_energy_schedule(
            1000,
            energy_budget_uj=1.0,
            energy_per_neuron_nj=100.0,
        )
        assert s.neurons_per_epoch == 10
        assert s.duty_cycle == 0.01

    def test_priority_neurons(self):
        from sc_neurocore.compiler.intelligence import (
            generate_energy_schedule,
        )

        s = generate_energy_schedule(
            100,
            priority_neurons=[50, 51, 52],
        )
        assert s.update_order[0] == 50
        assert s.update_order[1] == 51

    def test_excess_budget(self):
        from sc_neurocore.compiler.intelligence import (
            generate_energy_schedule,
        )

        s = generate_energy_schedule(
            10,
            energy_budget_uj=1000.0,
        )
        assert s.neurons_per_epoch == 10
        assert s.duty_cycle == 1.0

    def test_rejects_invalid_schedule_inputs(self):
        from sc_neurocore.compiler.intelligence import (
            generate_energy_schedule,
        )

        invalid_cases = [
            ({"neuron_count": 0}, "neuron_count"),
            ({"neuron_count": 10, "energy_budget_uj": -1.0}, "energy_budget_uj"),
            ({"neuron_count": 10, "energy_per_neuron_nj": 0.0}, "energy_per_neuron_nj"),
            ({"neuron_count": 10, "epoch_duration_ms": 0.0}, "epoch_duration_ms"),
            ({"neuron_count": 10, "priority_neurons": [-1]}, "priority_neurons"),
            ({"neuron_count": 10, "priority_neurons": [10]}, "priority_neurons"),
        ]
        for kwargs, message in invalid_cases:
            with pytest.raises(ValueError, match=message):
                generate_energy_schedule(**kwargs)

    def test_priority_neurons_are_deduplicated(self):
        from sc_neurocore.compiler.intelligence import (
            generate_energy_schedule,
        )

        s = generate_energy_schedule(5, priority_neurons=[2, 2, 1])
        assert s.update_order[:2] == [2, 1]
        assert len(s.update_order) == len(set(s.update_order))


class TestThermalEnvelope:
    def test_pass(self):
        from sc_neurocore.compiler.intelligence import (
            estimate_thermal_envelope,
        )

        t = estimate_thermal_envelope(power_mw=100, theta_ja=25)
        assert t.pass_fail == "PASS"
        assert t.t_junction == 27.5  # 25 + 0.1*25

    def test_fail(self):
        from sc_neurocore.compiler.intelligence import (
            estimate_thermal_envelope,
        )

        t = estimate_thermal_envelope(
            power_mw=5000,
            theta_ja=30,
            t_junction_max=100,
        )
        assert t.pass_fail == "FAIL"
        assert t.thermal_margin < 0

    def test_margin(self):
        from sc_neurocore.compiler.intelligence import (
            estimate_thermal_envelope,
        )

        t = estimate_thermal_envelope(power_mw=0)
        assert t.thermal_margin == 100.0  # 125 - 25


class TestWave7Integration:
    def test_classify_then_recommend(self):
        from sc_neurocore.compiler.intelligence import (
            classify_model_complexity,
            recommend_target,
        )

        m = classify_model_complexity({"v": "a * b + c * d - e"})
        recs = recommend_target(
            {"v": "a * b + c * d - e"},
            require_class=m.recommended_paradigm,
        )
        assert len(recs) > 0

    def test_recommend_then_risk(self):
        from sc_neurocore.compiler.intelligence import (
            recommend_target,
            score_supply_chain_risk,
        )

        recs = recommend_target({"v": "a + b"}, top_n=1)
        risk = score_supply_chain_risk(recs[0].profile_name)
        assert risk.risk_score >= 0

    def test_bittrue_then_thermal(self):
        from sc_neurocore.compiler.intelligence import (
            generate_bittrue_kernel,
            estimate_thermal_envelope,
        )

        code = generate_bittrue_kernel("sc_lif", {"v": "a + b"})
        assert len(code) > 50
        t = estimate_thermal_envelope(power_mw=50)
        assert t.pass_fail == "PASS"

    def test_cache_workflow(self):
        from sc_neurocore.compiler.intelligence import (
            CompilationCache,
            generate_bittrue_kernel,
        )

        cache = CompilationCache()
        eqs = {"v": "a + b"}
        assert cache.get(eqs, "artix7") is None
        code = generate_bittrue_kernel("sc_lif", eqs)
        cache.put(eqs, "artix7", 16, 8, {"code": code})
        hit = cache.get(eqs, "artix7")
        assert hit["code"] == code


class TestPowerIntent:
    def test_upf_output(self):
        from sc_neurocore.compiler.intelligence import generate_power_intent

        upf = generate_power_intent("sc_lif")
        assert "set_scope sc_lif" in upf
        assert "PD_NEURON_0" in upf
        assert "set_isolation" in upf

    def test_num_domains(self):
        from sc_neurocore.compiler.intelligence import generate_power_intent

        upf = generate_power_intent("sc_lif", num_domains=4)
        assert "PD_NEURON_3" in upf


class TestPowerFSM:
    def test_default(self):
        from sc_neurocore.compiler.intelligence import generate_power_state_machine

        v = generate_power_state_machine("sc_lif")
        assert "ACTIVE" in v
        assert "HIBERNATE" in v
        assert "power_fsm" in v

    def test_custom_states(self):
        from sc_neurocore.compiler.intelligence import generate_power_state_machine

        v = generate_power_state_machine("sc_lif", states=["ON", "OFF"])
        assert "ON" in v
        assert "OFF" in v


class TestThermalAnalysisBranches:
    """Cover the per-process-node derating tiers, the medium/low hotspot bands,
    and the finite-input guard that the existing thermal cases leave untouched."""

    def test_16nm_node_derates_less_than_7nm(self):
        from sc_neurocore.compiler.intelligence import thermal_analysis

        # 16 nm takes the >7, <=16 derating tier (x0.99), distinct from 7 nm (x0.98),
        # so with identical load the 16 nm result keeps a higher derated frequency.
        t16 = thermal_analysis(100.0, 500.0, process_nm=16)
        t7 = thermal_analysis(100.0, 500.0, process_nm=7)
        assert t16.derated_freq_mhz > t7.derated_freq_mhz

    def test_medium_hotspot_band(self):
        from sc_neurocore.compiler.intelligence import thermal_analysis

        # 150 muls across 10 columns -> 15 per column, inside the (10, 20] band.
        t = thermal_analysis(100.0, 500.0, mul_count=150, dsp_columns=10)
        assert t.hotspot_risk == "medium"

    def test_low_hotspot_band(self):
        from sc_neurocore.compiler.intelligence import thermal_analysis

        # 50 muls across 10 columns -> 5 per column, inside the (4, 10] band.
        t = thermal_analysis(100.0, 500.0, mul_count=50, dsp_columns=10)
        assert t.hotspot_risk == "low"

    def test_non_finite_ambient_is_rejected(self):
        from sc_neurocore.compiler.intelligence import thermal_analysis

        with pytest.raises(ValueError, match="t_ambient_c must be finite"):
            thermal_analysis(100.0, 500.0, t_ambient_c=float("nan"))
