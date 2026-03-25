# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# Tests for sc_neurocore.resilience (fault injection)

from __future__ import annotations

import numpy as np

from sc_neurocore.resilience import FaultResilienceSuite, FaultModel
from sc_neurocore.resilience.fault_suite import FaultType


def _eval_fn(weights):
    """Dummy eval: accuracy ~ mean absolute weight (bigger = better)."""
    return float(np.clip(np.mean([np.abs(w).mean() for w in weights]), 0, 1))


class TestFaultModel:
    def test_fields(self):
        fm = FaultModel(fault_type=FaultType.STUCK_AT_ZERO, rate=0.1)
        assert fm.rate == 0.1
        assert fm.layer_index is None


class TestFaultResilienceSuite:
    def _make_suite(self):
        weights = [np.random.randn(8, 4) * 0.5, np.random.randn(2, 8) * 0.5]
        return FaultResilienceSuite(eval_fn=_eval_fn, weights=weights)

    def test_baseline(self):
        suite = self._make_suite()
        assert suite.baseline_accuracy > 0

    def test_inject_stuck_at_zero(self):
        suite = self._make_suite()
        fault = FaultModel(FaultType.STUCK_AT_ZERO, rate=0.5, seed=42)
        faulted = suite.inject_fault(fault)
        zero_frac = np.mean(faulted[0] == 0)
        assert zero_frac > 0.3

    def test_inject_bit_flip(self):
        suite = self._make_suite()
        fault = FaultModel(FaultType.WEIGHT_BIT_FLIP, rate=0.5, seed=42)
        faulted = suite.inject_fault(fault)
        assert not np.array_equal(faulted[0], suite.weights[0])

    def test_inject_per_layer(self):
        suite = self._make_suite()
        fault = FaultModel(FaultType.DEAD_SYNAPSE, rate=0.9, layer_index=0, seed=42)
        faulted = suite.inject_fault(fault)
        # Layer 0 heavily faulted, layer 1 untouched
        assert np.mean(faulted[0] == 0) > 0.5
        np.testing.assert_array_equal(faulted[1], suite.weights[1])

    def test_run_single(self):
        suite = self._make_suite()
        fault = FaultModel(FaultType.STUCK_AT_ZERO, rate=0.1)
        result = suite.run_single(fault)
        assert result.accuracy_before >= result.accuracy_after
        assert result.degradation >= 0

    def test_sweep(self):
        suite = self._make_suite()
        report = suite.sweep(FaultType.STUCK_AT_ZERO, rates=[0.01, 0.1, 0.5])
        assert len(report.results) == 3
        curve = report.degradation_curve(FaultType.STUCK_AT_ZERO)
        assert len(curve) == 3

    def test_sweep_per_layer(self):
        suite = self._make_suite()
        report = suite.sweep(FaultType.DEAD_SYNAPSE, rates=[0.1, 0.5], per_layer=True)
        assert len(report.results) == 4  # 2 layers x 2 rates

    def test_full_audit(self):
        suite = self._make_suite()
        report = suite.full_audit()
        assert len(report.results) > 0
        s = report.summary()
        assert "Fault Resilience" in s

    def test_most_vulnerable_layer(self):
        suite = self._make_suite()
        report = suite.sweep(FaultType.STUCK_AT_ZERO, rates=[0.5], per_layer=True)
        mvl = report.most_vulnerable_layer()
        assert mvl in [0, 1]

    def test_noisy_membrane(self):
        suite = self._make_suite()
        fault = FaultModel(FaultType.NOISY_MEMBRANE, rate=0.1, seed=42)
        faulted = suite.inject_fault(fault)
        assert not np.array_equal(faulted[0], suite.weights[0])

    def test_bitstream_bias(self):
        suite = self._make_suite()
        fault = FaultModel(FaultType.BITSTREAM_BIAS, rate=0.3, seed=42)
        faulted = suite.inject_fault(fault)
        assert not np.array_equal(faulted[0], suite.weights[0])

    def test_stuck_at_one(self):
        suite = self._make_suite()
        fault = FaultModel(FaultType.STUCK_AT_ONE, rate=0.5, seed=42)
        faulted = suite.inject_fault(fault)
        assert np.any(faulted[0] == 1.0)
