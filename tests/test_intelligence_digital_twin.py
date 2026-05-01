# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore

import unittest

from sc_neurocore.compiler.intelligence import ingest_telemetry

class TestHILCalibration:
    def test_basic(self):
        from sc_neurocore.compiler.intelligence import generate_hil_calibration

        r = generate_hil_calibration("sc_lif", {"v": "expr", "u": "expr"})
        assert r.num_parameters == 2
        assert len(r.protocol_steps) >= 5

    def test_custom_ranges(self):
        from sc_neurocore.compiler.intelligence import generate_hil_calibration

        r = generate_hil_calibration(
            "sc_lif",
            {"v": "expr"},
            parameters={"tau": (-1.0, 1.0)},
        )
        assert r.sweep_ranges["tau"] == (-1.0, 1.0)


class TestDigitalTwin:
    def test_basic(self):
        from sc_neurocore.compiler.intelligence import generate_digital_twin

        code = generate_digital_twin("sc_lif", {"v": "-(v)/tau"}, "artix7")
        assert "Twin" in code
        assert "def step" in code
        assert "def compare" in code


class TestSEUScrubber:
    def test_leo(self):
        from sc_neurocore.compiler.intelligence import schedule_seu_scrubbing

        s = schedule_seu_scrubbing(1_000_000, orbit_altitude_km=400)
        assert s.interval_ms > 0
        assert s.frames_per_cycle > 0
        assert s.strategy == "hybrid"

    def test_higher_orbit(self):
        from sc_neurocore.compiler.intelligence import schedule_seu_scrubbing

        leo = schedule_seu_scrubbing(1_000_000, orbit_altitude_km=400)
        geo = schedule_seu_scrubbing(1_000_000, orbit_altitude_km=35786)
        # Higher orbit = more flux = shorter interval
        assert geo.interval_ms < leo.interval_ms


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

