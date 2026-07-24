# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestEligibilityTrace from former test_learning_advanced.py

"""Focused suite: TestEligibilityTrace from former test_learning_advanced.py."""

from __future__ import annotations

from tests.learning_advanced_support import *  # noqa: F403


class TestEligibilityTrace:
    def test_trace_starts_zero(self):
        et = EligibilityTrace(tau_e=20.0, dt=1.0)
        pre = np.zeros(5)
        post = np.zeros(3)
        error = np.ones(3)
        dw = et.update(pre, post, error)
        assert dw.shape == (5, 3)
        np.testing.assert_allclose(dw, 0.0)

    def test_trace_accumulates(self):
        et = EligibilityTrace(tau_e=20.0, dt=1.0)
        pre = np.array([1.0, 0.0, 0.0])
        post = np.array([0.0, 1.0])
        error = np.array([1.0, 1.0])
        dw1 = et.update(pre, post, error)
        # Pre=1 and post=1 should produce nonzero at [0,1]
        assert dw1[0, 1] > 0

    def test_trace_decays(self):
        et = EligibilityTrace(tau_e=10.0, dt=1.0)
        pre = np.array([1.0, 0.0])
        post = np.array([1.0])
        error = np.array([1.0])
        dw1 = et.update(pre, post, error)
        # Next step with no spikes — trace should decay
        dw2 = et.update(np.zeros(2), np.zeros(1), np.array([1.0]))
        assert abs(dw2[0, 0]) < abs(dw1[0, 0])

    def test_error_gating(self):
        """Zero error should produce zero weight delta."""
        et = EligibilityTrace(tau_e=20.0, dt=1.0)
        pre = np.array([1.0])
        post = np.array([1.0])
        dw = et.update(pre, post, np.array([0.0]))
        np.testing.assert_allclose(dw, 0.0)

    def test_decay_constant(self):
        et = EligibilityTrace(tau_e=20.0, dt=1.0)
        expected_decay = np.exp(-1.0 / 20.0)
        np.testing.assert_allclose(et.decay, expected_decay)
