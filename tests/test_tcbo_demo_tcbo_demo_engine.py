# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestTCBODemoEngine from former test_tcbo_demo.py

"""Focused suite: TestTCBODemoEngine from former test_tcbo_demo.py."""

from __future__ import annotations

from tests.tcbo_demo_support import *  # noqa: F403

class TestTCBODemoEngine(unittest.TestCase):
    def test_init(self):
        engine = TCBODemoEngine(N=16)
        self.assertEqual(engine.N, 16)
        self.assertFalse(engine.is_running)

    def test_get_scenarios(self):
        engine = TCBODemoEngine()
        s = engine.get_scenarios()
        self.assertEqual(len(s), 5)
        self.assertIn("healthy_awake", s)

    def test_start_scenario(self):
        engine = TCBODemoEngine()
        info = engine.start_scenario("healthy_awake")
        self.assertTrue(engine.is_running)
        self.assertEqual(info["scenario"], "healthy_awake")

    def test_invalid_scenario(self):
        engine = TCBODemoEngine()
        with self.assertRaises(ValueError):
            engine.start_scenario("nonexistent")

    def test_step(self):
        engine = TCBODemoEngine()
        engine.start_scenario("healthy_awake")
        snap = engine.step()
        self.assertIsInstance(snap, TCBODemoSnapshot)
        self.assertEqual(len(snap.phases), 16)

    def test_snapshot_serialization(self):
        engine = TCBODemoEngine()
        engine.start_scenario("healthy_awake")
        d = engine.step().to_dict()
        self.assertIn("p_h1", d)
        self.assertIn("phases", d)

    def test_get_state(self):
        engine = TCBODemoEngine()
        engine.start_scenario("healthy_awake")
        engine.step()
        state = engine.get_state()
        self.assertTrue(state["running"])

    def test_reset(self):
        engine = TCBODemoEngine()
        engine.start_scenario("healthy_awake")
        for _ in range(100):
            engine.step()
        engine.reset()
        self.assertFalse(engine.is_running)
        self.assertEqual(engine.p_h1, 0.0)

    def test_history(self):
        engine = TCBODemoEngine()
        engine.start_scenario("healthy_awake")
        for _ in range(50):
            engine.step()
        h = engine.get_history(10)
        self.assertEqual(len(h), 10)

    def test_run_scenario(self):
        engine = TCBODemoEngine()
        snaps = engine.run_scenario("healthy_awake", duration_s=1.0, subsample=100)
        self.assertGreater(len(snaps), 5)

    def test_all_scenarios_run(self):
        engine = TCBODemoEngine(seed=42)
        for name in ScenarioName:
            snaps = engine.run_scenario(name.value, duration_s=1.0, subsample=100)
            self.assertGreater(len(snaps), 0)

    def test_step_without_start_raises(self):
        engine = TCBODemoEngine()
        with self.assertRaises(RuntimeError):
            engine.step()
