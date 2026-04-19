# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Stochastic Doctor Tests

"""Tests for the stochastic doctor diagnostics module."""

from __future__ import annotations

import json
import unittest

import numpy as np

from sc_neurocore.stochastic_doctor.diagnostics import (
    AuditSeverity,
    DriftDetector,
    StochasticDoctor,
    compute_scc,
)


class TestSCC(unittest.TestCase):
    """SCC computation tests."""

    def test_identical_streams(self):
        a = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 0], dtype=np.uint8)
        scc = compute_scc(a, a)
        self.assertAlmostEqual(scc, 1.0, places=5)

    def test_anticorrelated_streams(self):
        a = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.uint8)
        b = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.uint8)
        scc = compute_scc(a, b)
        self.assertAlmostEqual(scc, -1.0, places=5)

    def test_independent_streams(self):
        rng = np.random.default_rng(42)
        a = rng.integers(0, 2, size=10000, dtype=np.uint8)
        b = rng.integers(0, 2, size=10000, dtype=np.uint8)
        scc = compute_scc(a, b)
        self.assertLess(abs(scc), 0.1)


class TestDriftDetector(unittest.TestCase):
    """Drift detector tests."""

    def test_stable_no_drift(self):
        dd = DriftDetector(alpha=0.1, threshold=0.3)
        for _ in range(100):
            self.assertFalse(dd.observe(0.0))

    def test_sustained_correlation_triggers_drift(self):
        dd = DriftDetector(alpha=0.1, threshold=0.3)
        for _ in range(50):
            dd.observe(0.9)
        self.assertTrue(dd.active)
        self.assertGreater(dd.ema, 0.3)

    def test_reset_clears_state(self):
        dd = DriftDetector(alpha=0.1, threshold=0.3)
        for _ in range(50):
            dd.observe(0.9)
        dd.reset()
        self.assertEqual(dd.ema, 0.0)
        self.assertFalse(dd.active)
        self.assertEqual(len(dd.history), 0)

    def test_negative_correlation_drift(self):
        dd = DriftDetector(alpha=0.1, threshold=0.3)
        for _ in range(50):
            dd.observe(-0.8)
        self.assertTrue(dd.active)

    def test_history_tracking(self):
        dd = DriftDetector(alpha=0.5, threshold=0.9)
        dd.observe(0.5)
        dd.observe(0.5)
        self.assertEqual(len(dd.history), 2)


class TestStochasticDoctor(unittest.TestCase):
    """Full doctor audit tests."""

    def setUp(self):
        self.doc = StochasticDoctor(
            correlation_threshold=0.3,
            critical_threshold=0.7,
        )

    def test_compute_correlation(self):
        rng = np.random.default_rng(42)
        a = rng.integers(0, 2, size=10000, dtype=np.uint8)
        b = rng.integers(0, 2, size=10000, dtype=np.uint8)
        scc = self.doc.compute_correlation(a, b)
        self.assertLess(abs(scc), 0.1)
        scc_same = self.doc.compute_correlation(a, a)
        self.assertGreater(scc_same, 0.9)

    def test_estimate_precision(self):
        bs = np.zeros(1000, dtype=np.uint8)
        bs[:500] = 1
        p, var = self.doc.estimate_precision(bs)
        self.assertAlmostEqual(p, 0.5, places=2)
        self.assertGreater(var, 0.0)
        # Theoretical: 0.5*0.5/1000 = 0.00025
        self.assertAlmostEqual(var, 0.00025, places=4)

    def test_compute_histogram(self):
        bs = np.ones(128, dtype=np.uint8)
        hist = self.doc.compute_histogram(bs, word_size=64)
        self.assertEqual(len(hist), 65)
        self.assertEqual(hist[64], 2)
        self.assertEqual(hist[:64].sum(), 0)

    def test_audit_layer_critical_correlation(self):
        a = np.zeros(1024, dtype=np.uint8)
        a[:512] = 1
        b = a.copy()
        c = np.random.default_rng(42).integers(0, 2, size=1024, dtype=np.uint8)
        report = self.doc.audit_layer("test_layer", np.stack([a, b, c]))
        self.assertEqual(report.layer, "test_layer")
        self.assertEqual(report.num_neurons, 3)
        self.assertEqual(report.stream_length, 1024)
        self.assertGreater(report.max_correlation, 0.9)
        self.assertEqual(report.status, AuditSeverity.CRITICAL)
        self.assertGreater(len(report.hot_neurons), 0)

    def test_audit_layer_ok(self):
        rng = np.random.default_rng(42)
        streams = rng.integers(0, 2, size=(3, 1024), dtype=np.uint8)
        report = self.doc.audit_layer("ok_layer", streams)
        self.assertEqual(report.status, AuditSeverity.OK)

    def test_report_json_serialization(self):
        a = np.zeros(128, dtype=np.uint8)
        a[:64] = 1
        report = self.doc.audit_layer("json_test", np.stack([a, a]))
        json_str = report.to_json()
        parsed = json.loads(json_str)
        self.assertEqual(parsed["layer"], "json_test")
        self.assertIn("status", parsed)
        self.assertIn("findings", parsed)

    def test_report_to_dict(self):
        a = np.zeros(64, dtype=np.uint8)
        a[:32] = 1
        report = self.doc.audit_layer("dict_test", np.stack([a, a]))
        d = report.to_dict()
        self.assertIsInstance(d, dict)
        self.assertEqual(d["layer"], "dict_test")

    def test_precision_in_report(self):
        rng = np.random.default_rng(42)
        streams = rng.integers(0, 2, size=(4, 2048), dtype=np.uint8)
        report = self.doc.audit_layer("prec_test", streams)
        self.assertGreater(report.mean_precision, 0.3)
        self.assertLess(report.mean_precision, 0.7)


if __name__ == "__main__":
    unittest.main()
