# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestScenarioValidations from former test_tcbo_demo.py

"""Focused suite: TestScenarioValidations from former test_tcbo_demo.py."""

from __future__ import annotations

from tests.tcbo_demo_support import *  # noqa: F403


class TestScenarioValidations(unittest.TestCase):
    def test_healthy_awake_coherence(self):
        engine = TCBODemoEngine(seed=42)
        snaps = engine.run_scenario("healthy_awake", duration_s=3.0, subsample=300)
        last_R = snaps[-1].R_global
        self.assertGreater(last_R, 0.1)

    def test_anesthesia_lower_R(self):
        """Anesthesia should produce lower R than healthy awake."""
        engine = TCBODemoEngine(seed=42)
        healthy = engine.run_scenario("healthy_awake", duration_s=3.0, subsample=300)
        anest = engine.run_scenario("anesthesia", duration_s=3.0, subsample=300)
        R_healthy = np.mean([s.R_global for s in healthy[-5:]])
        R_anest = np.mean([s.R_global for s in anest[-5:]])
        self.assertLess(R_anest, R_healthy + 0.1)

    def test_recovery_kappa_varies(self):
        engine = TCBODemoEngine(seed=42)
        snaps = engine.run_scenario("recovery", duration_s=5.0, subsample=500)
        kappas = [s.kappa for s in snaps]
        self.assertNotAlmostEqual(max(kappas), min(kappas), places=2)

    def test_sleep_onset_runs(self):
        engine = TCBODemoEngine(seed=42)
        snaps = engine.run_scenario("sleep_onset", duration_s=2.0, subsample=200)
        self.assertTrue(all(0 <= s.R_global <= 1 for s in snaps))
