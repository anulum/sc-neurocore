# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore

import unittest

import pytest

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

    def test_protocol_contains_design_matrix_and_acceptance_metadata(self):
        from sc_neurocore.compiler.intelligence import generate_hil_calibration

        r = generate_hil_calibration(
            "sc_lif",
            {"v": "-v/tau"},
            parameters={"tau": (5.0, 50.0), "threshold": (0.5, 2.0)},
            sample_points=5,
            repetitions=3,
            settle_cycles=16,
            acceptance_tolerance=1e-3,
            correction_model="weighted_least_squares",
        )

        assert r.sample_count == 15
        assert len(r.design_matrix) == 5
        assert {tuple(point) for point in r.design_matrix} == {("tau", "threshold")}
        assert r.observables == ("v",)
        assert r.correction_model == "weighted_least_squares"
        assert r.acceptance_tolerance == 1e-3
        assert any("settle 16 cycles" in step for step in r.protocol_steps)
        assert any("weighted_least_squares" in step for step in r.protocol_steps)

    def test_rejects_invalid_calibration_contract(self):
        from sc_neurocore.compiler.intelligence import generate_hil_calibration

        invalid_cases = [
            dict(module_name="", equations={"v": "expr"}, parameters={"tau": (0.0, 1.0)}),
            dict(module_name="sc_lif", equations={}, parameters={"tau": (0.0, 1.0)}),
            dict(module_name="sc_lif", equations={"v": "expr"}, parameters={"tau": (1.0, 1.0)}),
            dict(module_name="sc_lif", equations={"v": "expr"}, sample_points=1),
            dict(module_name="sc_lif", equations={"v": "expr"}, repetitions=0),
            dict(module_name="sc_lif", equations={"v": "expr"}, acceptance_tolerance=0.0),
        ]

        for kwargs in invalid_cases:
            with pytest.raises(ValueError):
                generate_hil_calibration(**kwargs)


class TestHILCalibrationContractEdges:
    """The remaining validation branches of the HIL contract: each guards a
    distinct malformed sweep specification that the happy-path tests never reach."""

    def test_negative_settle_cycles_rejected(self):
        from sc_neurocore.compiler.intelligence import generate_hil_calibration

        with pytest.raises(ValueError, match="settle_cycles must be >= 0"):
            generate_hil_calibration("sc_lif", {"v": "expr"}, settle_cycles=-1)

    def test_parameter_with_wrong_bound_count_rejected(self):
        from sc_neurocore.compiler.intelligence import generate_hil_calibration

        with pytest.raises(ValueError, match="exactly two bounds"):
            generate_hil_calibration(
                "sc_lif",
                {"v": "expr"},
                parameters={"tau": (0.0, 1.0, 2.0)},
            )

    def test_non_finite_parameter_bounds_rejected(self):
        from sc_neurocore.compiler.intelligence import generate_hil_calibration

        with pytest.raises(ValueError, match="bounds must be finite"):
            generate_hil_calibration(
                "sc_lif",
                {"v": "expr"},
                parameters={"tau": (0.0, float("inf"))},
            )

    def test_empty_parameter_map_rejected(self):
        from sc_neurocore.compiler.intelligence import generate_hil_calibration

        # equations is non-empty (so the earlier guard passes) but the explicit
        # parameter map is empty, leaving no sweep range to calibrate.
        with pytest.raises(ValueError, match="at least one sweep range"):
            generate_hil_calibration("sc_lif", {"v": "expr"}, parameters={})

    def test_observable_absent_from_equations_rejected(self):
        from sc_neurocore.compiler.intelligence import generate_hil_calibration

        with pytest.raises(ValueError, match="not present in equations"):
            generate_hil_calibration(
                "sc_lif",
                {"v": "expr"},
                observables=("v", "phantom"),
            )

    def test_coprime_stride_advances_past_shared_factor(self):
        from sc_neurocore.compiler.intelligence import generate_hil_calibration

        # The second sweep dimension seeds its Latin-hypercube stride at 3; with
        # nine sample points gcd(3, 9) == 3, so the stride search must step past
        # the shared factor before the design matrix can be built.
        r = generate_hil_calibration(
            "sc_lif",
            {"a": "expr", "b": "expr"},
            parameters={"a": (0.0, 1.0), "b": (0.0, 1.0)},
            sample_points=9,
        )
        assert len(r.design_matrix) == 9
        assert r.num_parameters == 2


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


class TestSeuScrubberFallback:
    """No configuration bits means no expected upsets, so the scrub interval
    falls back to the daily cadence rather than dividing by zero."""

    def test_zero_config_bits_uses_daily_fallback_interval(self):
        from sc_neurocore.compiler.intelligence import schedule_seu_scrubbing

        s = schedule_seu_scrubbing(0, orbit_altitude_km=400)
        assert s.interval_ms == round(24.0 * 3_600_000, 2)
        assert s.expected_seu_rate == 0.0
