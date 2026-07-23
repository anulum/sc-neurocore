# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestTelemetry from former test_intelligence_digital_twin.py

"""Focused suite: TestTelemetry from former test_intelligence_digital_twin.py."""

from __future__ import annotations

from tests.intelligence_digital_twin_support import *  # noqa: F403

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
