# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestResilienceBenchmark from former test_fault_injection.py

"""Focused suite: TestResilienceBenchmark from former test_fault_injection.py."""

from __future__ import annotations

from fault_injection_support import *  # noqa: F403

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
