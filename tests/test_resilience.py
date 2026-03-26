# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for hardware fault resilience suite

from __future__ import annotations

import numpy as np

from sc_neurocore.resilience import FaultResilienceSuite, FaultModel
from sc_neurocore.resilience.fault_suite import FaultType, FaultResult


def _eval_fn(weights):
    """Dummy eval: accuracy ~ mean absolute weight (bigger = better)."""
    return float(np.clip(np.mean([np.abs(w).mean() for w in weights]), 0, 1))


class TestFaultModel:
    def test_fields(self):
        fm = FaultModel(fault_type=FaultType.STUCK_AT_ZERO, rate=0.1)
        assert fm.rate == 0.1
        assert fm.layer_index is None

    def test_all_fault_types_exist(self):
        expected = {
            "stuck_at_0",
            "stuck_at_1",
            "weight_bit_flip",
            "dead_synapse",
            "noisy_membrane",
            "bitstream_bias",
        }
        actual = {ft.value for ft in FaultType}
        assert actual == expected


class TestFaultResult:
    def test_degradation(self):
        r = FaultResult(
            fault_type=FaultType.STUCK_AT_ZERO,
            fault_rate=0.1,
            layer_index=None,
            accuracy_before=0.95,
            accuracy_after=0.80,
            degradation=0.15,
        )
        assert r.degradation == 0.15


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

    def test_inject_stuck_at_one(self):
        suite = self._make_suite()
        fault = FaultModel(FaultType.STUCK_AT_ONE, rate=0.5, seed=42)
        faulted = suite.inject_fault(fault)
        assert np.any(faulted[0] == 1.0)

    def test_inject_bit_flip(self):
        suite = self._make_suite()
        fault = FaultModel(FaultType.WEIGHT_BIT_FLIP, rate=0.5, seed=42)
        faulted = suite.inject_fault(fault)
        assert not np.array_equal(faulted[0], suite.weights[0])

    def test_inject_per_layer(self):
        suite = self._make_suite()
        fault = FaultModel(FaultType.DEAD_SYNAPSE, rate=0.9, layer_index=0, seed=42)
        faulted = suite.inject_fault(fault)
        assert np.mean(faulted[0] == 0) > 0.5
        np.testing.assert_array_equal(faulted[1], suite.weights[1])

    def test_inject_noisy_membrane(self):
        suite = self._make_suite()
        fault = FaultModel(FaultType.NOISY_MEMBRANE, rate=0.1, seed=42)
        faulted = suite.inject_fault(fault)
        assert not np.array_equal(faulted[0], suite.weights[0])

    def test_inject_bitstream_bias(self):
        suite = self._make_suite()
        fault = FaultModel(FaultType.BITSTREAM_BIAS, rate=0.3, seed=42)
        faulted = suite.inject_fault(fault)
        assert not np.array_equal(faulted[0], suite.weights[0])

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
        assert len(report.results) == 4

    def test_full_audit(self):
        suite = self._make_suite()
        report = suite.full_audit()
        assert len(report.results) == len(FaultType) * 2 * 4
        s = report.summary()
        assert "Fault Resilience" in s

    def test_most_vulnerable_layer(self):
        suite = self._make_suite()
        report = suite.sweep(FaultType.STUCK_AT_ZERO, rates=[0.5], per_layer=True)
        mvl = report.most_vulnerable_layer()
        assert mvl in [0, 1]
