# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Fault Injection Tests

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

import numpy as np
import pytest

from sc_neurocore.fault_injection.fault_injection import (
    FaultInjector,
    FaultModel,
    FaultInjectionResult,
    RadiationProfile,
    ResilienceBenchmark,
    ResilienceReport,
)


class TestRadiationProfiles(unittest.TestCase):
    def test_leo(self):
        p = RadiationProfile.leo()
        self.assertEqual(p.name, "LEO")
        self.assertGreater(p.ber, 0)

    def test_geo_higher_than_leo(self):
        self.assertGreater(RadiationProfile.geo().ber, RadiationProfile.leo().ber)

    def test_deep_space_highest(self):
        self.assertGreater(RadiationProfile.deep_space().ber, RadiationProfile.geo().ber)

    def test_terrestrial_lowest(self):
        self.assertLess(RadiationProfile.terrestrial().ber, RadiationProfile.leo().ber)


class TestFaultInjectionResult(unittest.TestCase):
    def test_probability_calculation(self):
        r = FaultInjectionResult(
            original_popcount=500,
            corrupted_popcount=480,
            bits_flipped=20,
            bitstream_length=1000,
        )
        self.assertAlmostEqual(r.probability_original, 0.5)
        self.assertAlmostEqual(r.probability_corrupted, 0.48)
        self.assertAlmostEqual(r.absolute_error, 0.02)

    def test_zero_length_safety(self):
        r = FaultInjectionResult(0, 0, 0, 0)
        self.assertEqual(r.probability_original, 0.0)
        self.assertEqual(r.absolute_error, 0.0)


class TestFaultInjector(unittest.TestCase):
    def setUp(self):
        self.injector = FaultInjector(seed=42)

    def test_bit_flip_at_high_ber(self):
        bs = np.ones(1000, dtype=np.uint8)
        corrupted, flipped = self.injector.inject(bs, FaultModel.BIT_FLIP, ber=0.5)
        self.assertGreater(flipped, 100)
        self.assertLess(flipped, 900)

    def test_bit_flip_at_zero_ber(self):
        bs = np.ones(100, dtype=np.uint8)
        corrupted, flipped = self.injector.inject(bs, FaultModel.BIT_FLIP, ber=0.0)
        self.assertEqual(flipped, 0)
        np.testing.assert_array_equal(bs, corrupted)

    def test_stuck_at_0(self):
        bs = np.ones(1000, dtype=np.uint8)
        corrupted, affected = self.injector.inject(bs, FaultModel.STUCK_AT_0, ber=0.1)
        self.assertGreater(affected, 0)
        self.assertEqual(int(np.sum(corrupted)), 1000 - affected)

    def test_stuck_at_1(self):
        bs = np.zeros(1000, dtype=np.uint8)
        corrupted, affected = self.injector.inject(bs, FaultModel.STUCK_AT_1, ber=0.1)
        self.assertGreater(affected, 0)
        self.assertEqual(int(np.sum(corrupted)), affected)

    def test_gaussian_noise(self):
        bs = np.ones(1000, dtype=np.uint8)
        corrupted, changed = self.injector.inject(bs, FaultModel.GAUSSIAN_NOISE, ber=0.3)
        self.assertGreater(changed, 0)

    def test_dropout(self):
        bs = np.ones(1000, dtype=np.uint8)
        corrupted, affected = self.injector.inject(bs, FaultModel.DROPOUT, ber=0.2)
        self.assertGreater(affected, 0)
        self.assertLess(int(np.sum(corrupted)), 1000)

    def test_deterministic_injection(self):
        bs = np.array([1, 0, 1, 1, 0], dtype=np.uint8)
        corrupted = self.injector.inject_at_positions(bs, [0, 4])
        expected = np.array([0, 0, 1, 1, 1], dtype=np.uint8)
        np.testing.assert_array_equal(corrupted, expected)

    def test_inject_preserves_length(self):
        bs = np.ones(512, dtype=np.uint8)
        corrupted, _ = self.injector.inject(bs, FaultModel.BIT_FLIP, ber=0.01)
        self.assertEqual(len(corrupted), 512)


class TestResilienceBenchmark(unittest.TestCase):
    def setUp(self):
        self.bench = ResilienceBenchmark(seed=0)

    def test_run_produces_report(self):
        report = self.bench.run(
            fault_model=FaultModel.BIT_FLIP,
            ber=0.01,
            bitstream_length=256,
            num_trials=100,
        )
        self.assertIsInstance(report, ResilienceReport)

    def test_mean_error_positive_at_nonzero_ber(self):
        report = self.bench.run(
            fault_model=FaultModel.BIT_FLIP,
            ber=0.1,
            bitstream_length=256,
            num_trials=100,
        )
        self.assertGreater(report.mean_error, 0)

    def test_zero_ber_zero_error(self):
        report = self.bench.run(
            fault_model=FaultModel.BIT_FLIP,
            ber=0.0,
            bitstream_length=256,
            num_trials=50,
        )
        self.assertAlmostEqual(report.mean_error, 0.0, places=6)

    def test_longer_bitstream_more_resilient(self):
        r_short = self.bench.run(
            fault_model=FaultModel.BIT_FLIP,
            ber=0.01,
            bitstream_length=64,
            num_trials=200,
        )
        r_long = self.bench.run(
            fault_model=FaultModel.BIT_FLIP,
            ber=0.01,
            bitstream_length=2048,
            num_trials=200,
        )
        # Longer bitstreams should have smaller mean error (SC inherent resilience)
        self.assertLess(r_long.mean_error, r_short.mean_error * 1.5)

    def test_p95_gte_mean(self):
        report = self.bench.run(
            fault_model=FaultModel.BIT_FLIP,
            ber=0.05,
            bitstream_length=256,
            num_trials=200,
        )
        self.assertGreaterEqual(report.p95_error, report.mean_error)

    def test_report_summary(self):
        report = self.bench.run(
            fault_model=FaultModel.BIT_FLIP,
            ber=0.01,
            bitstream_length=256,
            num_trials=50,
        )
        summary = report.summary()
        self.assertIn("Fault:", summary)
        self.assertIn("BER:", summary)
        self.assertIn("Mean Error:", summary)

    def test_sweep_ber(self):
        reports = self.bench.sweep_ber(
            fault_model=FaultModel.BIT_FLIP,
            ber_range=[0.001, 0.01, 0.1],
            bitstream_length=256,
            num_trials=50,
        )
        self.assertEqual(len(reports), 3)
        # Error should increase with BER
        self.assertLess(reports[0].mean_error, reports[2].mean_error)

    def test_wall_time_positive(self):
        report = self.bench.run(
            fault_model=FaultModel.BIT_FLIP,
            ber=0.01,
            bitstream_length=256,
            num_trials=10,
        )
        self.assertGreater(report.wall_time_ms, 0)


_VALID_REPORT = dict(
    fault_model="bit_flip",
    ber=0.1,
    bitstream_length=10,
    num_trials=5,
    mean_error=0.1,
    std_error=0.05,
    max_error=0.3,
    p95_error=0.2,
    p99_error=0.25,
    mean_bits_flipped=1.0,
    wall_time_ms=1.0,
)


def test_radiation_profile_rejects_non_numeric_ber():
    with pytest.raises(ValueError, match="ber must be"):
        RadiationProfile(name="bad", ber="high")  # type: ignore[arg-type]


def test_fault_injection_result_rejects_non_integer_field():
    with pytest.raises(ValueError, match="must be an integer"):
        FaultInjectionResult(
            original_popcount="x",  # type: ignore[arg-type]
            corrupted_popcount=0,
            bits_flipped=0,
            bitstream_length=10,
        )


def test_resilience_report_rejects_empty_fault_model():
    with pytest.raises(ValueError, match="fault_model must be a non-empty string"):
        ResilienceReport(**{**_VALID_REPORT, "fault_model": "   "})


def test_resilience_report_rejects_non_numeric_field():
    with pytest.raises(ValueError, match="ber must be numeric"):
        ResilienceReport(**{**_VALID_REPORT, "ber": "x"})


def test_resilience_report_rejects_non_finite_field():
    with pytest.raises(ValueError, match="mean_error must be finite"):
        ResilienceReport(**{**_VALID_REPORT, "mean_error": float("inf")})


def test_inject_rejects_non_numeric_ber():
    inj = FaultInjector(seed=0)
    with pytest.raises(ValueError, match="ber must be"):
        inj.inject(np.array([0, 1], dtype=np.uint8), FaultModel.BIT_FLIP, "x")  # type: ignore[arg-type]


def test_inject_gaussian_requires_numeric_bitstream():
    inj = FaultInjector(seed=0)
    with pytest.raises(ValueError, match="gaussian_noise requires numeric"):
        inj.inject(np.array(["a", "b"]), FaultModel.GAUSSIAN_NOISE, 0.1)


def test_inject_unsupported_fault_model_raises():
    # A FaultModel-typed object that matches none of the handled members reaches
    # the exhaustiveness guard (defended for forward compatibility / typing).
    inj = FaultInjector(seed=0)
    bogus = MagicMock(spec=FaultModel)
    with pytest.raises(ValueError, match="unsupported fault model"):
        inj.inject(np.array([0, 1], dtype=np.uint8), bogus, 0.5)


def test_inject_at_positions_rejects_non_array():
    inj = FaultInjector(seed=0)
    with pytest.raises(ValueError, match="must be a numpy.ndarray"):
        inj.inject_at_positions([0, 1, 0], [1])  # type: ignore[arg-type]


def test_inject_at_positions_rejects_non_1d():
    inj = FaultInjector(seed=0)
    with pytest.raises(ValueError, match="must be a 1-D array"):
        inj.inject_at_positions(np.zeros((2, 2), dtype=np.uint8), [0])


def test_generate_bitstream_rejects_non_numeric_probability():
    bench = ResilienceBenchmark(seed=0)
    with pytest.raises(ValueError, match="probability must be"):
        bench._generate_bitstream(8, "x")  # type: ignore[arg-type]


def test_run_rejects_non_numeric_probability():
    bench = ResilienceBenchmark(seed=0)
    with pytest.raises(ValueError, match="probability must be"):
        bench.run(fault_model=FaultModel.BIT_FLIP, ber=0.1, probability="x")  # type: ignore[arg-type]


def test_run_rejects_non_numeric_ber():
    bench = ResilienceBenchmark(seed=0)
    with pytest.raises(ValueError, match="ber must be"):
        bench.run(fault_model=FaultModel.BIT_FLIP, ber="x")  # type: ignore[arg-type]


def test_sweep_ber_rejects_non_fault_model():
    bench = ResilienceBenchmark(seed=0)
    with pytest.raises(ValueError, match="fault_model must be a FaultModel"):
        bench.sweep_ber(fault_model="bit_flip", ber_range=[0.1])  # type: ignore[arg-type]


def test_sweep_ber_rejects_non_numeric_entry():
    bench = ResilienceBenchmark(seed=0)
    with pytest.raises(ValueError, match="ber_range entries must be"):
        bench.sweep_ber(fault_model=FaultModel.BIT_FLIP, ber_range=["x"])  # type: ignore[list-item]


if __name__ == "__main__":
    unittest.main()
