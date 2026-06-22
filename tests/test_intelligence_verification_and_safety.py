# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore

import unittest

from sc_neurocore.compiler.intelligence import (
    configure_approximation,
    explore_pareto,
    generate_dvfs_controller,
    ingest_telemetry,
    model_energy_harvest,
    predict_aging,
    predict_reliability,
    protect_ip_pqc,
    run_fault_campaign,
    verify_timing_closure,
)
from sc_neurocore.compiler.platforms import get_profile


class TestAging(unittest.TestCase):
    def test_degradation_increases(self):
        r5 = predict_aging(250.0, years=5.0, temperature_c=25.0)
        r10 = predict_aging(250.0, years=10.0, temperature_c=25.0)
        self.assertGreater(r10.degradation_pct, r5.degradation_pct)
        self.assertLess(r10.degraded_fmax_mhz, r5.degraded_fmax_mhz)

    def test_high_temp_worse(self):
        r_cool = predict_aging(250.0, temperature_c=25.0)
        r_hot = predict_aging(250.0, temperature_c=125.0)
        self.assertGreater(r_hot.degradation_pct, r_cool.degradation_pct)

    def test_dominant_mechanism(self):
        r = predict_aging(250.0)
        self.assertIn(r.dominant_mechanism, ("NBTI", "HCI"))

    def test_high_voltage_stress_can_make_hci_dominant(self):
        r = predict_aging(250.0, voltage_v=1.4, temperature_c=25.0, years=10.0)
        self.assertGreater(r.hci_degradation_pct, r.nbti_degradation_pct)
        self.assertEqual(r.dominant_mechanism, "HCI")

    def test_predict_aging_rejects_invalid_inputs(self):
        invalid_cases = [
            ({"initial_fmax_mhz": 0.0}, "initial_fmax_mhz"),
            ({"initial_fmax_mhz": 250.0, "voltage_v": 0.0}, "voltage_v"),
            ({"initial_fmax_mhz": 250.0, "temperature_c": -274.0}, "temperature_c"),
            ({"initial_fmax_mhz": 250.0, "years": -1.0}, "years"),
        ]
        for kwargs, message in invalid_cases:
            with self.assertRaisesRegex(ValueError, message):
                predict_aging(**kwargs)


class TestReliability(unittest.TestCase):
    def test_predict_reliability_rejects_invalid_inputs(self):
        invalid_cases = [
            ({"voltage_v": 0.0}, "voltage_v"),
            ({"temperature_c": -274.0}, "temperature_c"),
            ({"node_nm": 0}, "node_nm"),
            ({"base_mttf_hours": 0.0}, "base_mttf_hours"),
        ]
        for kwargs, message in invalid_cases:
            with self.assertRaisesRegex(ValueError, message):
                predict_reliability(**kwargs)

    def test_predict_reliability_hotter_voltage_stressed_mttf_is_lower(self):
        nominal = predict_reliability(voltage_v=0.9, temperature_c=85.0)
        stressed = predict_reliability(voltage_v=1.05, temperature_c=105.0)
        self.assertLess(stressed.mttf_hours, nominal.mttf_hours)

    def test_predict_reliability_reports_per_mechanism_mttf(self):
        stressed = predict_reliability(voltage_v=1.3, temperature_c=105.0)
        self.assertIn("NBTI", stressed.mechanism_mttf_hours)
        self.assertIn("HCI", stressed.mechanism_mttf_hours)
        self.assertIn("TDDB", stressed.mechanism_mttf_hours)
        self.assertEqual(
            stressed.failure_mode,
            min(stressed.mechanism_mttf_hours, key=stressed.mechanism_mttf_hours.get),
        )


class TestFaultCampaign(unittest.TestCase):
    def test_basic(self):
        r = run_fault_campaign({"v": "a", "u": "b"})
        self.assertEqual(r.total_injections, 1000)
        self.assertGreater(r.sdc_count, 0)
        self.assertGreater(r.sdc_rate, 0)
        self.assertLess(r.sdc_rate, 1.0)

    def test_deterministic(self):
        r1 = run_fault_campaign({"v": "a"}, seed=123)
        r2 = run_fault_campaign({"v": "a"}, seed=123)
        self.assertEqual(r1.sdc_count, r2.sdc_count)


class TestTimingClosure(unittest.TestCase):
    def test_simple_passes(self):
        r = verify_timing_closure({"v": "-(v)/tau + I"}, target_freq_mhz=100.0)
        self.assertTrue(r.timing_met)
        self.assertGreater(r.slack_ns, 0)

    def test_complex_fails(self):
        eqs = {"v": "a*b*c*d*e*f*g*h + i + j + k + l"}
        r = verify_timing_closure(eqs, target_freq_mhz=2000.0, data_width=32)
        if not r.timing_met:
            self.assertGreater(len(r.recommendations), 0)


class TestWave11Integration(unittest.TestCase):
    def test_full_pipeline(self):
        """End-to-end: profile → approx → aging → timing → telemetry."""
        p = get_profile("extropic_epu")
        eqs = {"v": "-(v)/tau + I"}

        approx = configure_approximation(eqs)
        self.assertGreater(approx.total_energy_savings_pct, 0)

        aging = predict_aging(200.0, years=5.0)
        self.assertLess(aging.degraded_fmax_mhz, 200.0)

        timing = verify_timing_closure(eqs, target_freq_mhz=100.0)
        self.assertTrue(timing.timing_met)

        hw = [{"v": 1.0}]
        tw = [{"v": 1.0}]
        tel = ingest_telemetry(hw, tw)
        self.assertTrue(tel.healthy)

    def test_batteryless_pipeline(self):
        """Energy harvest → DVFS → Pareto → PQC."""
        harvest = model_energy_harvest(50.0, harvester_type="solar", environment="outdoor")
        self.assertTrue(harvest.energy_positive)

        dvfs = generate_dvfs_controller("sc_sensor")
        self.assertIn("endmodule", dvfs)

        pareto = explore_pareto({"v": "-(v)/tau"})
        self.assertGreater(len(pareto), 0)

        pqc = protect_ip_pqc("sc_sensor", {"v": "a"})
        self.assertTrue(pqc.quantum_safe)


class TestFormalEquivalence:
    """Formal equivalence proof skeleton."""

    def test_basic_sketch(self):
        from sc_neurocore.compiler.intelligence import (
            generate_equivalence_sketch,
        )

        s = generate_equivalence_sketch(
            "sc_lif",
            {"v": "a + b * c"},
        )
        assert s.module_name == "sc_lif"
        assert len(s.proof_steps) >= 5
        assert len(s.assertions) == 1
        assert s.quantisation_bound > 0

    def test_multi_equation(self):
        from sc_neurocore.compiler.intelligence import (
            generate_equivalence_sketch,
        )

        s = generate_equivalence_sketch(
            "sc_izh",
            {"v": "a * b + c", "u": "d * e"},
        )
        assert len(s.assertions) == 2
        assert "CONCLUSION" in s.proof_steps[-1]

    def test_sva_format(self):
        from sc_neurocore.compiler.intelligence import (
            generate_equivalence_sketch,
        )

        s = generate_equivalence_sketch("sc_lif", {"v": "a + b"})
        assert "assert property" in s.assertions[0]
        assert "posedge clk" in s.assertions[0]


class TestComplianceMatrix:
    """Safety compliance matrix generation."""

    def test_default_standards(self):
        from sc_neurocore.compiler.intelligence import (
            generate_compliance_matrix,
        )

        entries = generate_compliance_matrix("sc_lif")
        standards = {e.standard for e in entries}
        assert "DO-254" in standards
        assert "IEC 61508" in standards
        assert "ISO 26262" in standards

    def test_all_covered(self):
        from sc_neurocore.compiler.intelligence import (
            generate_compliance_matrix,
        )

        entries = generate_compliance_matrix(
            "sc_lif",
            has_tmr=True,
            has_checksum=True,
            has_sva=True,
            has_provenance=True,
        )
        covered = [e for e in entries if e.status == "covered"]
        assert len(covered) == len(entries)

    def test_gaps_without_tmr(self):
        from sc_neurocore.compiler.intelligence import (
            generate_compliance_matrix,
        )

        entries = generate_compliance_matrix("sc_lif")
        gaps = [e for e in entries if e.status == "gap"]
        assert len(gaps) > 0

    def test_format_report(self):
        from sc_neurocore.compiler.intelligence import (
            generate_compliance_matrix,
            format_compliance_report,
        )

        entries = generate_compliance_matrix("sc_lif", has_tmr=True)
        report = format_compliance_report(entries)
        assert "Compliance Matrix" in report
        assert "DO-254" in report
        assert "✅" in report


class TestBittrueKernel:
    def test_c_kernel(self):
        from sc_neurocore.compiler.intelligence import (
            generate_bittrue_kernel,
        )

        code = generate_bittrue_kernel("sc_lif", {"v": "a + b"})
        assert "#include <stdint.h>" in code
        assert "sc_lif_state_t" in code
        assert "sat(" in code
        assert "fxmul(" in code

    def test_rust_kernel(self):
        from sc_neurocore.compiler.intelligence import (
            generate_bittrue_kernel,
        )

        code = generate_bittrue_kernel(
            "sc_lif",
            {"v": "a + b"},
            language="rust",
        )
        assert "pub struct" in code
        assert "fn sat" in code
        assert "clamp" in code

    def test_multi_var(self):
        from sc_neurocore.compiler.intelligence import (
            generate_bittrue_kernel,
        )

        code = generate_bittrue_kernel(
            "sc_izh",
            {"v": "a * b", "u": "c + d"},
        )
        assert "int16_t v;" in code
        assert "int16_t u;" in code


class TestODEStability:
    def test_stable(self):
        from sc_neurocore.compiler.intelligence import verify_ode_stability

        r = verify_ode_stability({"v": "a"}, dt=0.1)
        assert r.stable is True

    def test_unstable(self):
        from sc_neurocore.compiler.intelligence import verify_ode_stability

        r = verify_ode_stability(
            {"v": "a"},
            dt=100.0,
            time_constants={"v": 0.5},
        )
        assert r.stable is False

    def test_critical_dt(self):
        from sc_neurocore.compiler.intelligence import verify_ode_stability

        r = verify_ode_stability(
            {"v": "a"},
            dt=0.1,
            time_constants={"v": 10.0},
        )
        assert r.critical_dt == 20.0


class TestFaultTree:
    def test_basic(self):
        from sc_neurocore.compiler.intelligence import generate_fault_tree

        ft = generate_fault_tree("sc_lif", {"v": "a", "u": "b"})
        assert "SYSTEM_FAILURE" in ft.top_event
        assert len(ft.basic_events) >= 6  # 2 vars * 2 + 2 common
        assert len(ft.mcs) == len(ft.basic_events)

    def test_single_var(self):
        from sc_neurocore.compiler.intelligence import generate_fault_tree

        ft = generate_fault_tree("sc_lif", {"v": "a"})
        assert len(ft.basic_events) == 4  # 1 var * 2 + 2 common


class TestAutoTestbench:
    def test_cocotb(self):
        from sc_neurocore.compiler.intelligence import generate_testbench

        tb = generate_testbench("sc_lif", {"v": "a"})
        assert "import cocotb" in tb
        assert "test_sc_lif_reset" in tb

    def test_uvm(self):
        from sc_neurocore.compiler.intelligence import generate_testbench

        tb = generate_testbench("sc_lif", {"v": "a"}, framework="uvm")
        assert "uvm_test" in tb


class TestWave8Integration:
    def test_nir_to_stability(self):
        from sc_neurocore.compiler.intelligence import (
            import_nir_graph,
            verify_ode_stability,
        )

        g = import_nir_graph(
            {
                "nodes": {"n0": {"type": "LIF", "tau": 10}},
                "edges": [],
            }
        )
        r = verify_ode_stability(g.equations, dt=0.1)
        assert r.stable is True

    def test_carbon_vs_reliability(self):
        from sc_neurocore.compiler.intelligence import (
            estimate_carbon_footprint,
            predict_reliability,
        )

        c = estimate_carbon_footprint("artix7", power_mw=500)
        r = predict_reliability(voltage_v=0.9, temperature_c=85)
        assert c.total_5yr_kg_co2 > 0
        assert r.mttf_years > 0

    def test_fault_tree_then_testbench(self):
        from sc_neurocore.compiler.intelligence import (
            generate_fault_tree,
            generate_testbench,
        )

        ft = generate_fault_tree("sc_lif", {"v": "a"})
        tb = generate_testbench("sc_lif", {"v": "a"})
        assert len(ft.mcs) > 0
        assert len(tb) > 100


class TestCDC:
    def test_same_domain(self):
        from sc_neurocore.compiler.intelligence import analyze_cdc

        r = analyze_cdc({"v": "a + b", "u": "c"})
        assert r.safe is True

    def test_cross_domain(self):
        from sc_neurocore.compiler.intelligence import analyze_cdc

        r = analyze_cdc(
            {"v": "u + 1", "u": "v - 1"},
            clock_domains={"v": "clk_a", "u": "clk_b"},
        )
        assert r.total_crossings >= 2


class TestRegressionWatchdog:
    def test_no_regression(self):
        from sc_neurocore.compiler.intelligence import check_regression

        r = check_regression({"area": 100}, {"area": 102})
        assert r[0].regression is False

    def test_regression(self):
        from sc_neurocore.compiler.intelligence import check_regression

        r = check_regression({"area": 100}, {"area": 120})
        assert r[0].regression is True
        assert r[0].delta_pct == 20.0


class TestWave9Integration:
    def test_toml_to_report(self, tmp_path):
        from sc_neurocore.compiler.intelligence import (
            load_profiles_from_toml,
            generate_compilation_report,
        )

        toml = tmp_path / "e2e.toml"
        toml.write_text(
            "[[profile]]\n"
            'name = "e2e_custom"\n'
            'vendor = "E2EVendor"\n'
            'platform_class = "custom"\n'
            "data_width = 16\n"
            "fraction = 8\n"
        )
        load_profiles_from_toml(str(toml))
        md = generate_compilation_report("sc_lif", {"v": "a"}, "e2e_custom")
        assert "E2EVendor" in md

    def test_cdc_then_floorplan(self):
        from sc_neurocore.compiler.intelligence import (
            analyze_cdc,
            plan_multi_die_floorplan,
        )

        r = analyze_cdc({"v": "u", "u": "v"}, clock_domains={"v": "clk_a", "u": "clk_b"})
        assert r.total_crossings >= 2
        fp = plan_multi_die_floorplan({"region_a": 500, "region_b": 500})
        assert fp.total_dies >= 1


class TestMultiDieAndRegressionEdges:
    """Cover the floorplan overflow placement and the zero-baseline regression
    delta that the nominal cases leave untouched."""

    def test_oversized_block_forced_onto_last_die(self):
        from sc_neurocore.compiler.intelligence import plan_multi_die_floorplan

        # A block larger than any die's capacity cannot be placed in the first-fit
        # sweep, so it is forced onto the last die.
        fp = plan_multi_die_floorplan({"huge": 5000}, die_capacity=1000, num_dies=4)
        assert fp.die_assignment["huge"] == 3

    def test_zero_baseline_reports_zero_delta(self):
        from sc_neurocore.compiler.intelligence import check_regression

        # A zero baseline has no defined percentage change, so the delta is 0.
        checks = check_regression({"leak": 0.0}, {"leak": 5.0})
        leak = next(c for c in checks if c.metric == "leak")
        assert leak.delta_pct == 0.0
        assert leak.regression is False
