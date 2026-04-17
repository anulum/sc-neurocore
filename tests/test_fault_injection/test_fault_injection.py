# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Fault Injection Tests

from __future__ import annotations
import sys
import os

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "..",
        "src",
        "sc_neurocore",
        "fault_injection",
    ),
)

import unittest
import numpy as np
from fault_injection import (
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


if __name__ == "__main__":
    unittest.main()
