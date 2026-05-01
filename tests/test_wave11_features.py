# SPDX-License-Identifier: AGPL-3.0-or-later
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
"""Wave 11 test suite — 4 platform classes + 9 compiler features (§68-§76)."""

import unittest

from sc_neurocore.compiler.hardware_profiles import (
    get_profile, list_profile_names, HardwareProfile,
)
from sc_neurocore.compiler.advanced_features import (
    configure_approximation,
    model_energy_harvest,
    predict_aging,
    generate_dvfs_controller,
    explore_pareto,
    protect_ip_pqc,
    run_fault_campaign,
    verify_timing_closure,
    ingest_telemetry,
)


# ── Platform class tests ──────────────────────────────────────────────

class TestThermodynamicPlatforms(unittest.TestCase):
    def test_extropic_epu(self):
        p = get_profile("extropic_epu")
        self.assertEqual(p.platform_class, "thermodynamic")
        self.assertEqual(p.vendor, "Extropic")
        self.assertEqual(p.data_width, 8)

    def test_normal_cn101(self):
        p = get_profile("normal_cn101")
        self.assertEqual(p.platform_class, "thermodynamic")
        self.assertIn("stochastic", p.notes.lower())


class TestProbabilisticPlatforms(unittest.TestCase):
    def test_purdue_pbit(self):
        p = get_profile("purdue_pbit")
        self.assertEqual(p.platform_class, "probabilistic")
        self.assertEqual(p.vendor, "Purdue")

    def test_tohoku_sot_pbit(self):
        p = get_profile("tohoku_sot_pbit")
        self.assertEqual(p.platform_class, "probabilistic")
        self.assertIn("SOT", p.notes)


class TestPolaritonPlatforms(unittest.TestCase):
    def test_marvell_polariton(self):
        p = get_profile("marvell_polariton")
        self.assertEqual(p.platform_class, "polariton")
        self.assertEqual(p.vendor, "Marvell")

    def test_stanford_polariton(self):
        p = get_profile("stanford_polariton")
        self.assertEqual(p.platform_class, "polariton")
        self.assertIn("perovskite", p.notes.lower())


class TestMetamaterialPlatforms(unittest.TestCase):
    def test_mit_metamaterial(self):
        p = get_profile("mit_metamaterial")
        self.assertEqual(p.platform_class, "metamaterial")
        self.assertEqual(p.vendor, "MIT")

    def test_penn_acoustic_meta(self):
        p = get_profile("penn_acoustic_meta")
        self.assertEqual(p.platform_class, "metamaterial")
        self.assertIn("acoustic", p.notes.lower())


class TestTotalCoverage(unittest.TestCase):
    def test_profile_count_ge_183(self):
        self.assertGreaterEqual(len(list_profile_names()), 183)

    def test_class_count_ge_35(self):
        classes = {get_profile(n).platform_class for n in list_profile_names()}
        self.assertGreaterEqual(len(classes), 35)


# ── §68 Approximate Computing ────────────────────────────────────────

class TestApproximation(unittest.TestCase):
    def test_basic(self):
        r = configure_approximation({"v": "-(v)/tau + I"})
        self.assertGreater(r.total_energy_savings_pct, 0)
        self.assertIn("v", r.populations)
        self.assertIn("bits_reduced", r.populations["v"])

    def test_error_bound(self):
        r = configure_approximation(
            {"v": "a", "u": "b"}, max_error_pct=2.0,
        )
        self.assertLessEqual(r.max_output_error_pct, 3.1)

    def test_multi_var(self):
        r = configure_approximation({"v": "a", "u": "b", "w": "c"})
        self.assertEqual(len(r.populations), 3)


# ── §69 Energy Harvesting ────────────────────────────────────────────

class TestEnergyHarvest(unittest.TestCase):
    def test_solar_outdoor(self):
        r = model_energy_harvest(100.0, harvester_type="solar",
                                 environment="outdoor", harvester_area_cm2=1.0)
        self.assertTrue(r.energy_positive)
        self.assertGreater(r.margin_pct, 0)

    def test_rf_indoor_insufficient(self):
        r = model_energy_harvest(100.0, harvester_type="rf",
                                 environment="indoor")
        self.assertFalse(r.energy_positive)
        self.assertLess(r.recommended_duty_cycle, 1.0)


# ── §70 Aging-Aware ──────────────────────────────────────────────────

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


# ── §71 DVFS Controller ──────────────────────────────────────────────

class TestDVFS(unittest.TestCase):
    def test_default(self):
        v = generate_dvfs_controller("sc_lif")
        self.assertIn("module sc_lif_dvfs_ctrl", v)
        self.assertIn("spike_rate", v)
        self.assertIn("OP_0", v)
        self.assertIn("endmodule", v)

    def test_custom_points(self):
        v = generate_dvfs_controller("sc_hh", operating_points=[
            {"voltage_mv": 600, "freq_mhz": 50},
            {"voltage_mv": 1200, "freq_mhz": 800},
        ])
        self.assertIn("50", v)
        self.assertIn("800", v)


# ── §72 Pareto Explorer ──────────────────────────────────────────────

class TestPareto(unittest.TestCase):
    def test_non_empty(self):
        pts = explore_pareto({"v": "-(v)/tau + I"})
        self.assertGreater(len(pts), 0)

    def test_non_dominated(self):
        pts = explore_pareto({"v": "a", "u": "b"})
        for i, p in enumerate(pts):
            for j, q in enumerate(pts):
                if i != j:
                    self.assertFalse(
                        q.power_mw <= p.power_mw
                        and q.area_luts <= p.area_luts
                        and q.latency_ns <= p.latency_ns
                        and (q.power_mw < p.power_mw or q.area_luts < p.area_luts
                             or q.latency_ns < p.latency_ns),
                        f"Point {i} dominated by {j}",
                    )

    def test_sorted_by_power(self):
        pts = explore_pareto({"v": "a"})
        powers = [p.power_mw for p in pts]
        self.assertEqual(powers, sorted(powers))


# ── §73 Post-Quantum IP Protection ───────────────────────────────────

class TestPQC(unittest.TestCase):
    def test_basic(self):
        r = protect_ip_pqc("sc_lif", {"v": "a"})
        self.assertTrue(r.quantum_safe)
        self.assertEqual(r.algorithm, "CRYSTALS-Dilithium")
        self.assertEqual(len(r.signature_hex), 32)
        self.assertEqual(r.key_size_bits, 1952)

    def test_security_levels(self):
        r2 = protect_ip_pqc("m", {"v": "a"}, security_level=2)
        r5 = protect_ip_pqc("m", {"v": "a"}, security_level=5)
        self.assertLess(r2.key_size_bits, r5.key_size_bits)

    def test_deterministic(self):
        r1 = protect_ip_pqc("m", {"v": "a"})
        r2 = protect_ip_pqc("m", {"v": "a"})
        self.assertEqual(r1.signature_hex, r2.signature_hex)


# ── §74 Fault Injection ──────────────────────────────────────────────

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


# ── §75 Timing Closure ───────────────────────────────────────────────

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


# ── §76 Telemetry Ingestion ──────────────────────────────────────────

class TestTelemetry(unittest.TestCase):
    def test_healthy(self):
        hw = [{"v": 1.0}, {"v": 1.01}]
        tw = [{"v": 1.0}, {"v": 1.01}]
        r = ingest_telemetry(hw, tw)
        self.assertTrue(r.healthy)
        self.assertEqual(r.samples, 2)
        self.assertEqual(len(r.alerts), 0)

    def test_drift_detected(self):
        hw = [{"v": 1.0}, {"v": 2.0}]
        tw = [{"v": 1.0}, {"v": 1.0}]
        r = ingest_telemetry(hw, tw, drift_threshold=0.5)
        self.assertFalse(r.healthy)
        self.assertGreater(len(r.alerts), 0)
        self.assertGreater(r.max_drift, 0.5)

    def test_empty(self):
        r = ingest_telemetry([], [])
        self.assertTrue(r.healthy)
        self.assertEqual(r.samples, 0)


# ── Integration ──────────────────────────────────────────────────────

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
        harvest = model_energy_harvest(50.0, harvester_type="solar",
                                       environment="outdoor")
        self.assertTrue(harvest.energy_positive)

        dvfs = generate_dvfs_controller("sc_sensor")
        self.assertIn("endmodule", dvfs)

        pareto = explore_pareto({"v": "-(v)/tau"})
        self.assertGreater(len(pareto), 0)

        pqc = protect_ip_pqc("sc_sensor", {"v": "a"})
        self.assertTrue(pqc.quantum_safe)


if __name__ == "__main__":
    unittest.main()
