# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestWave11Integration from former test_intelligence_verification_and_safety.py

"""Focused suite: TestWave11Integration from former test_intelligence_verification_and_safety.py."""

from __future__ import annotations

from tests.intelligence_verification_and_safety_support import *  # noqa: F403

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
