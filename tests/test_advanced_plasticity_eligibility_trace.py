# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestEligibilityTrace from former test_advanced_plasticity.py

"""Focused suite: TestEligibilityTrace from former test_advanced_plasticity.py."""

from __future__ import annotations

from tests.advanced_plasticity_support import *  # noqa: F403

class TestEligibilityTrace:
    def test_output_shape(self):
        et = EligibilityTrace(tau_e=20.0)
        pre = np.array([1, 0, 1, 0, 0], dtype=np.float64)
        post = np.array([0, 1, 1], dtype=np.float64)
        err = np.array([0.5, -0.3, 0.1])
        delta = et.update(pre, post, err)
        assert delta.shape == (5, 3)

    def test_trace_decays(self):
        et = EligibilityTrace(tau_e=5.0)
        pre = np.array([1.0, 0.0])
        post = np.array([1.0])
        err = np.array([1.0])
        d1 = et.update(pre, post, err).copy()
        pre_zero = np.array([0.0, 0.0])
        d2 = et.update(pre_zero, post, err).copy()
        assert np.all(np.abs(d2) <= np.abs(d1) + 1e-12)
